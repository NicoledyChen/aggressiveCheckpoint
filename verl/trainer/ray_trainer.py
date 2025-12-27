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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import re
import subprocess
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import (
    CHECKPOINT_TRACKER,
    find_latest_ckpt,
    merge_and_prune_non_latest_actor_checkpoints,
    prune_non_latest_checkpoints_to_model_only,
    spawn_merge_and_prune_non_latest_actor_checkpoints,
    remove_obsolete_ckpt,
)
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloaders: dict[str, StatefulDataLoader],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self._bg_merge_procs: list[subprocess.Popen] = []

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

        if self.config.trainer.save_resume_state_only_latest and not self.config.trainer.save_model_only:
            prune_non_latest_checkpoints_to_model_only(
                self.config.trainer.save_checkpoint_path,
                latest_global_step=self.global_step,
            )

        if self.config.trainer.merge_sharded_model_to_hf_on_prune:
            # Only merge non-latest checkpoints so the latest checkpoint can still be resumed from.
            if self.config.trainer.merge_sharded_model_async:
                # clean finished background processes
                self._bg_merge_procs = [p for p in self._bg_merge_procs if p.poll() is None]
                available = max(0, self.config.trainer.merge_sharded_model_max_concurrent - len(self._bg_merge_procs))
                if available > 0:
                    max_to_process = min(self.config.trainer.merge_sharded_model_max_per_save, available)
                    new_procs = spawn_merge_and_prune_non_latest_actor_checkpoints(
                        self.config.trainer.save_checkpoint_path,
                        latest_global_step=self.global_step,
                        max_to_process=max_to_process,
                        nice=self.config.trainer.merge_sharded_model_background_nice,
                    )
                    self._bg_merge_procs.extend(new_procs)
            else:
                merge_and_prune_non_latest_actor_checkpoints(
                    self.config.trainer.save_checkpoint_path,
                    latest_global_step=self.global_step,
                    max_to_process=self.config.trainer.merge_sharded_model_max_per_save,
                )

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        def _normalize_answer_for_vote(ans: str) -> str:
            # lightweight normalization borrowed from dapo reward
            substitutions = [
                ("an ", ""),
                ("a ", ""),
                (".$", "$"),
                ("$", ""),
                (" ", ""),
                (r"\ ", ""),
                (",\\text{and}", ","),
                ("\\text{and}", ","),
            ]
            removed = [
                "square",
                "ways",
                "integers",
                "dollars",
                "mph",
                "inches",
                "hours",
                "km",
                "units",
                "\\ldots",
                "points",
                "feet",
                "minutes",
                "digits",
                "cents",
                "degrees",
                "cm",
                "gm",
                "pounds",
                "meters",
                "meals",
                "edges",
                "students",
                "childrentickets",
                "multiples",
                "\\text{s}",
                "\\text{.}",
                "\\text{\n}",
                "\\text{}^2",
                "\\text{}^3",
                "\\text{}",
            ]
            for b, a in substitutions:
                ans = ans.replace(b, a)
            for r in removed:
                ans = ans.replace(r, "")
            return ans.strip()

        def _extract_answer(text: str) -> str:
            match = re.findall(r"(?i)answer\s*:\s*([^\n]+)", text)
            cand = match[-1] if match else text
            return _normalize_answer_for_vote(cand)

        def _save_jsonl(path: str, rows: list[dict[str, Any]]):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        def _compute_passk(records_by_sample: dict[str, list[dict[str, Any]]], ks: list[int]):
            results = {}
            for k in ks:
                hits = []
                for sample_id, recs in records_by_sample.items():
                    topk = recs[: min(k, len(recs))]
                    hits.append(any(r["is_correct"] for r in topk))
                results[k] = float(np.mean(hits)) if hits else 0.0
            return results

        def _compute_self_consistency(records_by_sample: dict[str, list[dict[str, Any]]]):
            sc_hits = []
            for _, recs in records_by_sample.items():
                if not recs:
                    continue
                vote = Counter([r["normalized_answer"] for r in recs])
                majority_answer, _ = vote.most_common(1)[0]
                # correctness judged by any normalized answer equals majority AND is correct
                majority_correct = any(
                    (r["normalized_answer"] == majority_answer) and r["is_correct"] for r in recs
                )
                sc_hits.append(majority_correct)
            return float(np.mean(sc_hits)) if sc_hits else 0.0

        def _build_records(dataset_name: str, override_meta: dict[str, Any]):
            reward_tensor_lst = []
            reward_metrics_lst = defaultdict(list)
            length_metrics_lst = defaultdict(list)
            records: list[dict[str, Any]] = []
            samples_for_table = {"inputs": [], "outputs": [], "labels": [], "scores": []}

            loader = self.val_dataloaders[dataset_name]
            for batch_dict in loader:
                test_batch = DataProto.from_single_dict(batch_dict)
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
                repeat_times = override_meta.get("n", 1)
                test_gen_batch.meta_info = override_meta
                test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
                test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
                test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

                test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

                test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
                test_batch = test_batch.union(test_output_gen_batch)

                reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                output_ids = test_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                scores = reward_tensor.sum(-1).cpu().tolist()

                sample_ids = test_batch.non_tensor_batch.get("sample_id", np.arange(len(test_batch)))
                gts = test_batch.non_tensor_batch["ground_truth"]
                accuracies = reward_metrics.get("accuracy", [0.0] * len(test_batch))

                for idx in range(len(test_batch)):
                    rec = {
                        "sample_id": str(sample_ids[idx]),
                        "ground_truth": str(gts[idx]),
                        "response": output_texts[idx],
                        "normalized_answer": _extract_answer(output_texts[idx]),
                        "reward": scores[idx],
                        "accuracy": float(accuracies[idx]),
                        "is_correct": float(accuracies[idx]) > 0,
                        "step": self.global_step,
                    }
                    records.append(rec)

                samples_for_table["inputs"].extend(input_texts)
                samples_for_table["outputs"].extend(output_texts)
                samples_for_table["labels"].extend(gts.tolist())
                samples_for_table["scores"].extend(scores)

                reward_tensor_lst.append(reward_tensor)
                for key, value in reward_metrics.items():
                    reward_metrics_lst[key].extend(value)

                for key, value in compute_length_metrics(test_batch).items():
                    length_metrics_lst[key].append(value)

            return {
                "records": records,
                "reward_tensor": torch.cat(reward_tensor_lst, dim=0) if reward_tensor_lst else None,
                "reward_metrics_lst": reward_metrics_lst,
                "length_metrics_lst": length_metrics_lst,
                "samples_for_table": samples_for_table,
            }

        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        val_results: dict[str, Any] = {}

        for dataset_name in self.val_dataloaders:
            # high-temp / main sampling for pass@k and logging
            main_override = deepcopy(self.config.worker.rollout.val_override_config)
            data = _build_records(dataset_name, override_meta=main_override)
            records = data["records"]
            reward_tensor_cat = data["reward_tensor"]
            reward_metrics_lst = data["reward_metrics_lst"]
            length_metrics_lst = data["length_metrics_lst"]
            samples_for_table = data["samples_for_table"]

            records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in records:
                records_by_sample[rec["sample_id"]].append(rec)

            passk_vals = _compute_passk(records_by_sample, ks=[1, 8, 32])
            sc_acc = _compute_self_consistency(records_by_sample)
            sc_gain = sc_acc - passk_vals.get(1, 0.0)

            # Save per-step answer distribution plot (e.g. for AIME).
            if self.config.trainer.save_answer_distribution_plots:
                try:
                    pattern = re.compile(self.config.trainer.answer_distribution_plot_dataset_regex)
                    match_name = bool(pattern.search(dataset_name))
                    match_val_path = bool(
                        dataset_name == "val" and pattern.search(str(self.config.data.val_files or ""))
                    )
                    if match_name or match_val_path:
                        from ..utils.answer_distribution_plot import save_answer_distribution_grid_png

                        out_dir = os.path.join(
                            self.config.trainer.save_checkpoint_path,
                            "val_cache",
                            dataset_name,
                            "answer_dist",
                        )
                        out_path = os.path.join(out_dir, f"answer_dist_step_{self.global_step}.png")
                        save_answer_distribution_grid_png(
                            records_by_sample,
                            out_path,
                            top_k=self.config.trainer.answer_distribution_plot_top_k,
                            max_questions=self.config.trainer.answer_distribution_plot_max_questions,
                            title=f"{dataset_name} answer distribution @ step {self.global_step}",
                        )
                except Exception as e:
                    print(f"Failed to save answer distribution plot for {dataset_name}: {e}")

            # greedy pass@1 (temp=0, n=1) for the same dataset
            greedy_override = deepcopy(main_override)
            greedy_override["temperature"] = 0.0
            greedy_override["top_p"] = 1.0
            greedy_override["n"] = 1
            greedy_data = _build_records(dataset_name, override_meta=greedy_override)
            greedy_records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rec in greedy_data["records"]:
                greedy_records_by_sample[rec["sample_id"]].append(rec)
            pass1_greedy = _compute_passk(greedy_records_by_sample, ks=[1]).get(1, 0.0)

            # cache per-step records
            cache_dir = os.path.join(self.config.trainer.save_checkpoint_path, "val_cache", dataset_name)
            matrix_path = os.path.join(cache_dir, f"val_matrix_step_{self.global_step}.jsonl")
            _save_jsonl(matrix_path, records)

            # ensemble over cached steps
            ensemble_metrics = self._compute_ensembles(cache_dir, dataset_name)

            self._maybe_log_val_generations(
                samples_for_table["inputs"],
                samples_for_table["outputs"],
                samples_for_table["labels"],
                samples_for_table["scores"],
            )

            reward_score = reward_tensor_cat.sum(-1).mean().item() if reward_tensor_cat is not None else 0.0
            val_results.update(
                {
                    f"val/{dataset_name}/reward_score": reward_score,
                    **{f"val/{dataset_name}/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()},
                    **{f"val/{dataset_name}/{key}": value for key, value in reduce_metrics(length_metrics_lst).items()},
                    f"val/{dataset_name}/pass@1": pass1_greedy,
                    f"val/{dataset_name}/pass@8": passk_vals[8],
                    f"val/{dataset_name}/pass@32": passk_vals[32],
                    f"val/{dataset_name}/self_consistency_acc": sc_acc,
                    f"val/{dataset_name}/self_consistency_gain": sc_gain,
                    **{f"val/{dataset_name}/ensemble/{k}": v for k, v in ensemble_metrics.items()},
                }
            )
            # track best on primary val set
            if dataset_name == "val":
                self.val_reward_score = reward_score

        self.actor_rollout_ref_wg.release_rollout_engine()
        print("Finish validation.")
        return val_results

    def _compute_ensembles(self, cache_dir: str, dataset_name: str) -> dict[str, float]:
        if not os.path.isdir(cache_dir):
            return {}
        files = [f for f in os.listdir(cache_dir) if f.startswith("val_matrix_step_") and f.endswith(".jsonl")]
        if not files:
            return {}
        def _step_from_name(name: str) -> int:
            try:
                return int(name.split("val_matrix_step_")[1].split(".jsonl")[0])
            except Exception:
                return -1
        files = sorted(files, key=_step_from_name)
        max_files = 10  # limit for efficiency
        files = files[-max_files:]

        history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fname in files:
            step = _step_from_name(fname)
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    rec["step"] = step
                    history[rec["sample_id"]].append(rec)

        strategies = {}

        def eval_majority(records_subset: dict[str, list[dict[str, Any]]]) -> float:
            hits = []
            for recs in records_subset.values():
                vote = Counter([r["normalized_answer"] for r in recs])
                majority, _ = vote.most_common(1)[0]
                majority_correct = any((r["normalized_answer"] == majority) and r["is_correct"] for r in recs)
                hits.append(majority_correct)
            return float(np.mean(hits)) if hits else 0.0

        # full history majority
        strategies["majority_all"] = eval_majority(history)

        # reward-weighted majority (sum reward per answer)
        hits = []
        for recs in history.values():
            reward_sum = defaultdict(float)
            for r in recs:
                reward_sum[r["normalized_answer"]] += r.get("reward", 0.0)
            best_answer = max(reward_sum.items(), key=lambda x: x[1])[0]
            correct = any((r["normalized_answer"] == best_answer) and r["is_correct"] for r in recs)
            hits.append(correct)
        strategies["reward_weighted_all"] = float(np.mean(hits)) if hits else 0.0

        # recent-k majorities
        for k in (3, 5):
            recent_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for sid, recs in history.items():
                recent = sorted(recs, key=lambda x: x["step"], reverse=True)[:k]
                recent_history[sid] = recent
            strategies[f"majority_recent_{k}"] = eval_majority(recent_history)

        # greedy best subset of steps (up to 5) by majority vote
        unique_steps = sorted({rec["step"] for recs in history.values() for rec in recs})
        unique_steps = unique_steps[-5:]  # limit search space
        best_acc = -1.0
        best_subset = []
        from itertools import combinations
        for r in range(1, len(unique_steps) + 1):
            for subset in combinations(unique_steps, r):
                subset_history = defaultdict(list)
                for sid, recs in history.items():
                    subset_history[sid] = [rec for rec in recs if rec["step"] in subset]
                acc = eval_majority(subset_history)
                if acc > best_acc:
                    best_acc = acc
                    best_subset = list(subset)
        if best_subset:
            strategies["majority_best_subset_acc"] = best_acc
            strategies["majority_best_subset_size"] = float(len(best_subset))

        return strategies

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            if "entropy_loss" in metrics:
                metrics.setdefault("train/entropy", metrics["entropy_loss"])

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
