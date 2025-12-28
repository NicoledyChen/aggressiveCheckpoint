#!/usr/bin/env python3
"""
Build an EasyR1 aggregation dataset (problem/answer/solutions[]) from temporal-forgetting style
folders that contain `model_final_answer_{DATASET}.json`.

Expected layout (example):
  sampling_64_responses/
    Qwen__Qwen2.5-7B/
      model_final_answer_AIME.json
      model_final_answer_AIME25.json
      model_final_answer_AMC.json
    UWNSL__Qwen2.5-7B-deepscaler_4k_step_32/
      model_final_answer_AIME.json
      ...

Each `model_final_answer_*.json` is a JSON list of questions:
  {
    "index": 0,
    "problem": "...",
    "answer": "204",
    "responses": [
      {"resp_idx": 0, "resp_answer": "204", ...},
      ...
    ],
    ...
  }

We create one output row per question:
  {
    "problem": "...",
    "answer": "...",
    "solutions": ["Answer: ...", ...],
    "sample_id": "AIME:0"
  }

This dataset can be used with:
  - examples/format_prompt/agg_math.jinja
  - examples/reward_function/agg_math.py
  - examples/config_agg_math.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Ckpt:
    name: str
    path: str
    step: Optional[int]


def _parse_step(name: str) -> Optional[int]:
    m = re.search(r"_step_(\d+)", name)
    return int(m.group(1)) if m else None


def _list_checkpoint_dirs(root_dir: str, include_regex: Optional[str]) -> list[Ckpt]:
    pat = re.compile(include_regex) if include_regex else None
    ckpts: list[Ckpt] = []
    for entry in os.listdir(root_dir):
        full = os.path.join(root_dir, entry)
        if not os.path.isdir(full):
            continue
        if pat and (not pat.search(entry)):
            continue
        ckpts.append(Ckpt(name=entry, path=full, step=_parse_step(entry)))

    # Sort by step if present, otherwise keep stable name order.
    ckpts.sort(key=lambda c: (c.step is None, c.step if c.step is not None else 0, c.name))
    return ckpts


def _load_model_final_answer(ckpt_dir: str, dataset_name: str) -> Optional[list[dict[str, Any]]]:
    path = os.path.join(ckpt_dir, f"model_final_answer_{dataset_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _pick_answers(
    responses: list[dict[str, Any]],
    k: int,
    pick: str,
    rng: random.Random,
    unique: bool,
) -> list[str]:
    answers = []
    for r in responses:
        ans = _coerce_str(r.get("resp_answer", "")).strip()
        # Filter obvious placeholders / empty artifacts.
        if not ans:
            continue
        if "{{" in ans and "}}" in ans:
            continue
        if not ans:
            continue
        answers.append(ans)

    if unique:
        seen = set()
        uniq = []
        for a in answers:
            if a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        answers = uniq

    if k <= 0:
        return []
    if k >= len(answers):
        # return all (optionally shuffled)
        out = list(answers)
        if pick == "random":
            rng.shuffle(out)
        return out

    if pick == "first":
        return answers[:k]
    if pick == "random":
        return rng.sample(answers, k)

    raise ValueError(f"Unknown pick strategy: {pick}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Path to sampling_64_responses/")
    ap.add_argument("--dataset", required=True, choices=["AIME", "AIME25", "AMC", "Olympiad"])
    ap.add_argument(
        "--split_mode",
        choices=["train_val", "all"],
        default="train_val",
        help="train_val: write *_train.jsonl and *_val.jsonl. all: write a single *.jsonl containing all rows.",
    )
    ap.add_argument("--include_ckpt_regex", default=None, help="Regex to filter checkpoint directory names.")
    ap.add_argument("--k_per_ckpt", type=int, default=1, help="How many candidates to take from each checkpoint.")
    ap.add_argument("--pick", choices=["first", "random"], default="random")
    ap.add_argument("--unique_within_ckpt", action="store_true", help="Deduplicate answers within each checkpoint first.")
    ap.add_argument(
        "--dedup_global",
        action="store_true",
        help="Deduplicate solutions across checkpoints (after pooling).",
    )
    ap.add_argument("--max_solutions", type=int, default=16, help="Cap total solutions per sample (after pooling).")
    ap.add_argument("--shuffle_solutions", action="store_true", help="Shuffle pooled solutions before truncation.")
    ap.add_argument("--include_ckpt_name", action="store_true", help="Prefix each candidate with the checkpoint name.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--prefix", default="agg", help="Output file prefix.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ckpts = _list_checkpoint_dirs(args.root_dir, args.include_ckpt_regex)
    if not ckpts:
        raise RuntimeError(f"No checkpoint directories found under: {args.root_dir}")

    # Load per-ckpt data
    ckpt_data: list[tuple[Ckpt, dict[int, dict[str, Any]]]] = []
    for ckpt in ckpts:
        data = _load_model_final_answer(ckpt.path, args.dataset)
        if data is None:
            continue
        by_index: dict[int, dict[str, Any]] = {}
        for row in data:
            idx = row.get("index")
            if idx is None:
                continue
            by_index[int(idx)] = row
        if by_index:
            ckpt_data.append((ckpt, by_index))

    if not ckpt_data:
        raise RuntimeError(f"No model_final_answer_{args.dataset}.json found under any checkpoint dir.")

    # Compute intersection of available indices
    common: Optional[set[int]] = None
    for _, m in ckpt_data:
        s = set(m.keys())
        common = s if common is None else (common & s)
    indices = sorted(common or [])
    if not indices:
        raise RuntimeError("No common question indices across checkpoints.")

    # Build rows
    rows = []
    base_ckpt, base_map = ckpt_data[0]
    for idx in indices:
        base = base_map[idx]
        problem = _coerce_str(base.get("problem", "")).strip()
        answer = _coerce_str(base.get("answer", "")).strip()
        if not problem or answer == "":
            continue

        solutions: list[str] = []
        for ckpt, m in ckpt_data:
            item = m[idx]
            picked = _pick_answers(
                responses=item.get("responses", []) or [],
                k=args.k_per_ckpt,
                pick=args.pick,
                rng=rng,
                unique=args.unique_within_ckpt,
            )
            for a in picked:
                if args.include_ckpt_name:
                    solutions.append(f"[{ckpt.name}] Answer: {a}")
                else:
                    solutions.append(f"Answer: {a}")

        if args.dedup_global:
            seen = set()
            uniq = []
            for s in solutions:
                key = s.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            solutions = uniq

        if args.shuffle_solutions:
            rng.shuffle(solutions)

        if args.max_solutions > 0:
            solutions = solutions[: args.max_solutions]

        if not solutions:
            continue

        rows.append(
            {
                "problem": problem,
                "answer": answer,
                "solutions": solutions,
                "sample_id": f"{args.dataset}:{idx}",
                "meta": {
                    "dataset": args.dataset,
                    "num_ckpts": len(ckpt_data),
                    "base_ckpt": base_ckpt.name,
                },
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)

    def _write(path: str, xs: list[dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    rng.shuffle(rows)
    n_total = len(rows)
    if args.split_mode == "all":
        all_path = os.path.join(args.out_dir, f"{args.prefix}_{args.dataset}.jsonl")
        _write(all_path, rows)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "ckpts_used": [c.name for c, _ in ckpt_data],
                    "num_ckpts_used": len(ckpt_data),
                    "num_rows_total": n_total,
                    "split_mode": "all",
                    "out_all": os.path.abspath(all_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # Split train/val
    n_val = int(round(n_total * args.val_ratio))
    n_val = max(1, min(n_val, n_total - 1)) if n_total >= 2 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_path = os.path.join(args.out_dir, f"{args.prefix}_{args.dataset}_train.jsonl")
    val_path = os.path.join(args.out_dir, f"{args.prefix}_{args.dataset}_val.jsonl")
    _write(train_path, train_rows)
    _write(val_path, val_rows)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "ckpts_used": [c.name for c, _ in ckpt_data],
                "num_ckpts_used": len(ckpt_data),
                "num_rows_total": n_total,
                "num_rows_train": len(train_rows),
                "num_rows_val": len(val_rows),
                "split_mode": "train_val",
                "out_train": os.path.abspath(train_path),
                "out_val": os.path.abspath(val_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()



