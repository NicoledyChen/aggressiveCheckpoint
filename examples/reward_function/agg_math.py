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

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer

# Metadata
REWARD_NAME = "agg_math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    # Keep the same output contract as the vanilla math reward:
    # <think> ... </think> ... \boxed{...}
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _extract_chosen_candidate(response: str) -> Optional[int]:
    # Expected line (case-insensitive):
    #   Chosen candidate: <id|none>
    matches = re.findall(r"(?im)^\s*chosen\s*candidate\s*:\s*([^\n]+)\s*$", response)
    if not matches:
        return None
    raw = matches[-1].strip().lower()
    if raw in {"none", "null", "nil", "no", "n/a"}:
        return None
    m = re.search(r"\d+", raw)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _extract_candidate_answer(candidate_text: str) -> str:
    # Prefer explicit "Answer:" lines.
    m = re.findall(r"(?i)answer\s*:\s*([^\n]+)", candidate_text)
    if m:
        return m[-1].strip()
    # Fallback to boxed content if present.
    try:
        boxed = extract_boxed_content(candidate_text)
        if boxed is not None:
            boxed = str(boxed).strip()
            if boxed:
                return boxed
    except Exception:
        pass
    return candidate_text.strip()


def _selection_reward(
    response: str, ground_truth: str, solutions: Any
) -> tuple[float, dict[str, float]]:
    chosen = _extract_chosen_candidate(response)

    # Normalize solutions to a list of strings
    sol_list: list[str] = []
    if isinstance(solutions, list):
        sol_list = [str(s) for s in solutions]
    elif solutions is None:
        sol_list = []
    else:
        # allow numpy arrays / tuples etc.
        try:
            sol_list = [str(s) for s in list(solutions)]
        except Exception:
            sol_list = []

    correct_ids: list[int] = []
    for i, s in enumerate(sol_list, start=1):
        ans = _extract_candidate_answer(s)
        try:
            ok = grade_answer(ans, ground_truth)
        except Exception:
            ok = False
        if ok:
            correct_ids.append(i)

    has_correct = len(correct_ids) > 0
    chosen_valid = float(chosen is None or (1 <= int(chosen) <= len(sol_list)))

    if has_correct:
        selection = 1.0 if (chosen is not None and int(chosen) in correct_ids) else 0.0
    else:
        # If there is no correct trajectory among sampled candidates, the model should say "none".
        selection = 1.0 if chosen is None else 0.0

    metrics = {
        "selection": selection,
        "has_correct": float(has_correct),
        "chosen_valid": chosen_valid,
        "num_candidates": float(len(sol_list)),
        "num_correct_candidates": float(len(correct_ids)),
    }
    return selection, metrics


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    selection_weight: float = 0.2,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        # handle qwen2.5vl-32b format spacing like "< think >"
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        selection_score, selection_metrics = _selection_reward(
            response=response,
            ground_truth=reward_input["ground_truth"],
            solutions=reward_input.get("solutions"),
        )

        # weights
        fw = max(0.0, float(format_weight))
        sw = max(0.0, float(selection_weight))
        aw = max(0.0, 1.0 - fw - sw)
        scores.append(
            {
                "overall": aw * accuracy_score + fw * format_score + sw * selection_score,
                "format": format_score,
                "accuracy": accuracy_score,
                **selection_metrics,
            }
        )

    return scores


