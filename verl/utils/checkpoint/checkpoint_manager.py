# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin


CHECKPOINT_TRACKER = "checkpoint_tracker.json"
MERGE_IN_PROGRESS_FILE = "merge_in_progress.lock"


class BaseCheckpointManager(ABC):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        assert isinstance(self.model, FSDP)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def local_mkdir(path: str) -> str:
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)  # even if the lock is not acquired, try to create the directory

        return path

    @staticmethod
    def get_rng_state() -> dict[str, Any]:
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state: dict[str, Any]):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


def find_latest_ckpt(
    path: str, directory_format: str = "global_step_{}"
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Find the latest checkpoint in the save path.
    """
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        return None, None

    with open(tracker_file, "rb") as f:
        checkpointer_tracker_info = json.load(f)

    ckpt_path = os.path.join(path, directory_format.format(checkpointer_tracker_info["last_global_step"]))
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint does not exist: {ckpt_path}")
        return None, None

    print(f"Found latest checkpoint: {ckpt_path}, will resume from it. Turn off `find_last_checkpoint` to disable it.")
    return ckpt_path, checkpointer_tracker_info


def remove_obsolete_ckpt(
    path: str, global_step: int, best_global_step: int, save_limit: int = -1, directory_format: str = "global_step_{}"
):
    """
    Remove the obsolete checkpoints that exceed the save limit.
    """
    if save_limit <= 0 or not os.path.exists(path):
        return

    num_ckpt_to_keep = save_limit - 1  # exclude the current ckpt
    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    ckpt_global_steps = []
    for folder in os.listdir(path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step < global_step:
                ckpt_global_steps.append(step)

    ckpt_global_steps.sort(reverse=True)
    if best_global_step in ckpt_global_steps:  # do not remove the best ckpt
        ckpt_global_steps.remove(best_global_step)
        num_ckpt_to_keep = max(num_ckpt_to_keep - 1, 0)

    for step in ckpt_global_steps[num_ckpt_to_keep:]:
        folder_path = os.path.join(path, directory_format.format(step))
        lock_path = os.path.join(folder_path, MERGE_IN_PROGRESS_FILE)
        if os.path.exists(lock_path):
            print(f"Skip removing checkpoint (merge in progress): {folder_path}")
            continue
        try:
            shutil.rmtree(folder_path, ignore_errors=True)
            print(f"Removed obsolete checkpoint: {folder_path}")
        except Exception as e:
            print(f"Failed to remove {folder_path}: {e}")


def prune_checkpoint_to_model_only(
    ckpt_path: str,
    *,
    remove_dataloader_state: bool = True,
    role_subdirs: tuple[str, ...] = ("actor", "critic"),
) -> None:
    """Prune a checkpoint directory to keep model weights only.

    This is useful when you want to keep historical checkpoints for evaluation but only keep
    resume-critical states (optimizer/lr_scheduler/rng/dataloader) for the latest checkpoint.

    What will be removed (if exists):
    - `{ckpt_path}/dataloader.pt` (optional)
    - `{ckpt_path}/{actor,critic}/optim_world_size_*_rank_*.pt`
    - `{ckpt_path}/{actor,critic}/extra_state_world_size_*_rank_*.pt`

    What will be kept:
    - `{ckpt_path}/{actor,critic}/model_world_size_*_rank_*.pt`
    - `{ckpt_path}/{actor,critic}/huggingface/` (rank-0 saved)
    - any other files not matching the above patterns
    """

    if not ckpt_path or not os.path.isdir(ckpt_path):
        return

    if remove_dataloader_state:
        dataloader_path = os.path.join(ckpt_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            try:
                os.remove(dataloader_path)
                print(f"Pruned dataloader state: {dataloader_path}")
            except Exception as e:
                print(f"Failed to prune dataloader state {dataloader_path}: {e}")

    for role in role_subdirs:
        role_path = os.path.join(ckpt_path, role)
        if not os.path.isdir(role_path):
            continue

        try:
            filenames = os.listdir(role_path)
        except Exception as e:
            print(f"Failed to list {role_path}: {e}")
            continue

        for filename in filenames:
            if not filename.endswith(".pt"):
                continue

            if filename.startswith("optim_world_size_") or filename.startswith("extra_state_world_size_"):
                file_path = os.path.join(role_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Pruned resume state: {file_path}")
                except Exception as e:
                    print(f"Failed to prune {file_path}: {e}")


def prune_non_latest_checkpoints_to_model_only(
    root_path: str,
    latest_global_step: int,
    *,
    directory_format: str = "global_step_{}",
    remove_dataloader_state: bool = True,
    role_subdirs: tuple[str, ...] = ("actor", "critic"),
) -> None:
    """Prune all non-latest checkpoints under `root_path` to model-only.

    This keeps only the latest checkpoint (by `latest_global_step`) fully resumable.
    """

    if not root_path or not os.path.isdir(root_path):
        return

    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    for folder in os.listdir(root_path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step == latest_global_step:
                continue

            ckpt_path = os.path.join(root_path, folder)
            prune_checkpoint_to_model_only(
                ckpt_path,
                remove_dataloader_state=remove_dataloader_state,
                role_subdirs=role_subdirs,
            )


def _huggingface_has_model_weights(hf_path: str) -> bool:
    """Return True if the huggingface directory appears to contain model weights."""
    if not hf_path or not os.path.isdir(hf_path):
        return False

    try:
        for name in os.listdir(hf_path):
            # transformers may save either:
            # - pytorch_model.bin
            # - model.safetensors
            # - sharded files like model-00001-of-000xx.safetensors
            if name.endswith(".safetensors") or name.endswith(".bin"):
                return True
    except Exception:
        return False

    return False


def _has_sharded_model_files(local_dir: str) -> bool:
    if not local_dir or not os.path.isdir(local_dir):
        return False
    try:
        for name in os.listdir(local_dir):
            if re.match(r"model_world_size_\d+_rank_\d+\.pt$", name):
                return True
    except Exception:
        return False
    return False


def prune_sharded_model_files(local_dir: str) -> None:
    """Remove `model_world_size_*_rank_*.pt` files under a checkpoint role directory (e.g. actor/)."""
    if not local_dir or not os.path.isdir(local_dir):
        return
    try:
        for name in os.listdir(local_dir):
            if re.match(r"model_world_size_\d+_rank_\d+\.pt$", name):
                file_path = os.path.join(local_dir, name)
                try:
                    os.remove(file_path)
                    print(f"Pruned sharded model file: {file_path}")
                except Exception as e:
                    print(f"Failed to prune sharded model file {file_path}: {e}")
    except Exception as e:
        print(f"Failed to list {local_dir}: {e}")


def merge_sharded_model_to_hf(local_dir: str, *, timeout_s: Optional[int] = None) -> bool:
    """Merge sharded model checkpoint under `local_dir` into HuggingFace weights under `local_dir/huggingface`.

    This uses `scripts/model_merger.py` shipped with this repo.
    Returns True on success.
    """
    if not local_dir or not os.path.isdir(local_dir):
        return False

    # If already merged, skip.
    hf_path = os.path.join(local_dir, "huggingface")
    if _huggingface_has_model_weights(hf_path):
        return True

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/model_merger.py"))
    if not os.path.exists(script_path):
        print(f"model_merger script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path, "--local_dir", os.path.abspath(local_dir)]
    print(f"Running model merge: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        print(f"Model merge timed out for {local_dir}: {e}")
        return False
    except Exception as e:
        print(f"Model merge failed for {local_dir}: {e}")
        return False

    # Verify weights exist now
    if not _huggingface_has_model_weights(hf_path):
        print(f"Model merge finished but no HF weights found under {hf_path}")
        return False

    return True


def spawn_merge_and_prune_non_latest_actor_checkpoints(
    root_path: str,
    latest_global_step: int,
    *,
    directory_format: str = "global_step_{}",
    max_to_process: int = 1,
    nice: int = 10,
) -> list[subprocess.Popen]:
    """Spawn background processes to merge+prune sharded actor checkpoints.

    This starts `scripts/model_merger.py --delete_shards` in background so training isn't blocked.
    A lock file `{ckpt}/{MERGE_IN_PROGRESS_FILE}` is created to prevent the checkpoint from being deleted
    while merging; the merger script will remove the lock on exit.
    """

    procs: list[subprocess.Popen] = []
    if max_to_process <= 0:
        return procs
    if not root_path or not os.path.isdir(root_path):
        return procs

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/model_merger.py"))
    if not os.path.exists(script_path):
        print(f"model_merger script not found: {script_path}")
        return procs

    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    steps: list[int] = []
    for folder in os.listdir(root_path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step != latest_global_step:
                steps.append(step)

    steps.sort(reverse=True)
    processed = 0
    for step in steps:
        if processed >= max_to_process:
            break

        ckpt_path = os.path.join(root_path, directory_format.format(step))
        actor_dir = os.path.join(ckpt_path, "actor")
        if not _has_sharded_model_files(actor_dir):
            continue

        # If already merged (weights present), just prune shards synchronously.
        hf_path = os.path.join(actor_dir, "huggingface")
        if _huggingface_has_model_weights(hf_path):
            prune_sharded_model_files(actor_dir)
            continue

        lock_path = os.path.join(ckpt_path, MERGE_IN_PROGRESS_FILE)
        if os.path.exists(lock_path):
            continue

        # Create lock file to prevent deletion during merge
        try:
            with open(lock_path, "w") as f:
                f.write("")
        except Exception as e:
            print(f"Failed to create merge lock file {lock_path}: {e}")
            continue

        cmd = [
            sys.executable,
            script_path,
            "--local_dir",
            os.path.abspath(actor_dir),
            "--delete_shards",
            "--lock_file",
            os.path.abspath(lock_path),
        ]

        # Lower priority on unix-like systems if possible
        preexec_fn = None
        if hasattr(os, "nice") and isinstance(nice, int):
            def _set_nice():  # type: ignore
                try:
                    os.nice(nice)
                except Exception:
                    pass

            preexec_fn = _set_nice

        log_path = os.path.join(actor_dir, "merge.log")
        try:
            log_f = open(log_path, "a")
        except Exception:
            log_f = subprocess.DEVNULL  # type: ignore

        try:
            p = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=log_f,
                start_new_session=True,
                preexec_fn=preexec_fn,
            )
            procs.append(p)
            processed += 1
        except Exception as e:
            print(f"Failed to spawn merge process for {actor_dir}: {e}")
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass

    return procs


def merge_and_prune_non_latest_actor_checkpoints(
    root_path: str,
    latest_global_step: int,
    *,
    directory_format: str = "global_step_{}",
    max_to_process: int = 1,
    timeout_s: Optional[int] = None,
) -> None:
    """For non-latest checkpoints, merge actor sharded weights into HF format and delete sharded model files.

    This is intended to reduce file count/storage complexity for historical checkpoints while keeping the latest
    checkpoint resumable (latest keeps sharded model files).
    """
    if max_to_process <= 0:
        return
    if not root_path or not os.path.isdir(root_path):
        return

    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    steps: list[int] = []
    for folder in os.listdir(root_path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step != latest_global_step:
                steps.append(step)

    # process newest non-latest first (usually just the previous saved checkpoint)
    steps.sort(reverse=True)
    processed = 0
    for step in steps:
        if processed >= max_to_process:
            break

        ckpt_path = os.path.join(root_path, directory_format.format(step))
        actor_dir = os.path.join(ckpt_path, "actor")
        if not _has_sharded_model_files(actor_dir):
            continue

        ok = merge_sharded_model_to_hf(actor_dir, timeout_s=timeout_s)
        if not ok:
            continue

        prune_sharded_model_files(actor_dir)
        processed += 1
