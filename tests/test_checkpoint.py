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
import shutil
import uuid

import pytest

from verl.utils.checkpoint import (
    CHECKPOINT_TRACKER,
    find_latest_ckpt,
    prune_non_latest_checkpoints_to_model_only,
    remove_obsolete_ckpt,
)


@pytest.fixture
def save_checkpoint_path():
    ckpt_dir = os.path.join("checkpoints", str(uuid.uuid4()))
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_find_latest_ckpt(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"last_global_step": 10}, f, ensure_ascii=False, indent=2)

    assert find_latest_ckpt(save_checkpoint_path)[0] is None
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_10"), exist_ok=True)
    assert find_latest_ckpt(save_checkpoint_path)[0] == os.path.join(save_checkpoint_path, "global_step_10")


def test_remove_obsolete_ckpt(save_checkpoint_path):
    for step in range(5, 30, 5):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"), exist_ok=True)

    remove_obsolete_ckpt(save_checkpoint_path, global_step=30, best_global_step=10, save_limit=3)
    for step in range(5, 30, 5):
        is_exist = step in [10, 25]
        assert os.path.exists(os.path.join(save_checkpoint_path, f"global_step_{step}")) == is_exist


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


def test_prune_non_latest_checkpoints_to_model_only(save_checkpoint_path):
    # Create two checkpoints with dummy resume states
    for step in (1, 2):
        step_dir = os.path.join(save_checkpoint_path, f"global_step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        _touch(os.path.join(step_dir, "dataloader.pt"))

        for role in ("actor", "critic"):
            role_dir = os.path.join(step_dir, role)
            os.makedirs(os.path.join(role_dir, "huggingface"), exist_ok=True)
            _touch(os.path.join(role_dir, "huggingface", "config.json"))

            _touch(os.path.join(role_dir, "model_world_size_2_rank_0.pt"))
            _touch(os.path.join(role_dir, "optim_world_size_2_rank_0.pt"))
            _touch(os.path.join(role_dir, "extra_state_world_size_2_rank_0.pt"))

    prune_non_latest_checkpoints_to_model_only(save_checkpoint_path, latest_global_step=2)

    # step=1 should be pruned to model-only
    assert not os.path.exists(os.path.join(save_checkpoint_path, "global_step_1", "dataloader.pt"))
    assert os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "actor", "model_world_size_2_rank_0.pt")
    )
    assert not os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "actor", "optim_world_size_2_rank_0.pt")
    )
    assert not os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "actor", "extra_state_world_size_2_rank_0.pt")
    )
    assert os.path.exists(os.path.join(save_checkpoint_path, "global_step_1", "actor", "huggingface", "config.json"))

    assert os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "critic", "model_world_size_2_rank_0.pt")
    )
    assert not os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "critic", "optim_world_size_2_rank_0.pt")
    )
    assert not os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_1", "critic", "extra_state_world_size_2_rank_0.pt")
    )
    assert os.path.exists(os.path.join(save_checkpoint_path, "global_step_1", "critic", "huggingface", "config.json"))

    # step=2 (latest) should keep resume states
    assert os.path.exists(os.path.join(save_checkpoint_path, "global_step_2", "dataloader.pt"))
    assert os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_2", "actor", "optim_world_size_2_rank_0.pt")
    )
    assert os.path.exists(
        os.path.join(save_checkpoint_path, "global_step_2", "actor", "extra_state_world_size_2_rank_0.pt")
    )
