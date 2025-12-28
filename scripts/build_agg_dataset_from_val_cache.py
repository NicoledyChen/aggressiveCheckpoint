#!/usr/bin/env python3
"""
Build an AggLM-style aggregation dataset from EasyR1 validation caches.

EasyR1 (verl) writes validation sampling records to:
  {save_checkpoint_path}/val_cache/{dataset_name}/val_matrix_step_{step}.jsonl

Each line contains at least:
  - sample_id (string)
  - response (string)
  - reward (float, optional)
  - accuracy (float, optional)

This script joins those responses back to the original dataset rows to create a JSONL
compatible with EasyR1's RLHFDataset:
  - prompt_key (e.g. "problem")
  - answer_key (e.g. "answer")
  - solutions: list[str]
  - sample_id: stable id

Example:
  python3 scripts/build_agg_dataset_from_val_cache.py \
    --dataset hiyouga/math12k@test \
    --prompt_key problem \
    --answer_key answer \
    --cache_dir checkpoints/easy_r1/exp_name/val_cache/val \
    --max_step_files 10 \
    --max_candidates 8 \
    --per_step_pick best_reward \
    --out /tmp/agg_math_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Optional

from datasets import load_dataset


def _load_base_dataset(data_path: str, split: str) -> Any:
    """
    Mirror RLHFDataset loading semantics:
      - local dir (dataset builder) OR local file OR remote HF dataset name
    """
    if os.path.isdir(data_path):
        file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
        return load_dataset(file_type, data_dir=data_path, split=split)
    if os.path.isfile(data_path):
        file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
        return load_dataset(file_type, data_files=data_path, split=split)
    return load_dataset(data_path, split=split)


def _iter_val_matrix_files(cache_dir: str) -> list[tuple[int, str]]:
    """
    Return list of (step, file_path) sorted by step asc.
    """
    step_re = re.compile(r"^val_matrix_step_(\d+)\.jsonl$")
    files: list[tuple[int, str]] = []
    for name in os.listdir(cache_dir):
        m = step_re.match(name)
        if not m:
            continue
        step = int(m.group(1))
        files.append((step, os.path.join(cache_dir, name)))
    files.sort(key=lambda x: x[0])
    return files


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='Base dataset path. Supports "name@split" like RLHFDataset.')
    ap.add_argument("--prompt_key", default="problem")
    ap.add_argument("--answer_key", default="answer")
    ap.add_argument(
        "--id_key",
        default="sample_id",
        help="If this key exists in the base dataset, use it as stable id; otherwise use row index.",
    )
    ap.add_argument("--cache_dir", required=True, help="Directory containing val_matrix_step_*.jsonl")
    ap.add_argument("--max_step_files", type=int, default=10, help="Use only the most recent N step files.")
    ap.add_argument("--max_candidates", type=int, default=8, help="Max solutions per sample in output.")
    ap.add_argument(
        "--per_step_pick",
        choices=["best_reward", "best_accuracy", "random", "first"],
        default="best_reward",
        help="How to pick 1 candidate per sample per step file.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--keep_empty",
        action="store_true",
        help="Keep samples even if no candidates are found (solutions=[]). Not recommended for RL training.",
    )
    args = ap.parse_args()

    # Parse dataset@split
    if "@" in args.dataset:
        dataset_path, split = args.dataset.split("@", 1)
    else:
        dataset_path, split = args.dataset, "train"

    rng = random.Random(args.seed)

    if not os.path.isdir(args.cache_dir):
        raise FileNotFoundError(f"cache_dir not found or not a directory: {args.cache_dir}")

    step_files = _iter_val_matrix_files(args.cache_dir)
    if not step_files:
        raise FileNotFoundError(f"No val_matrix_step_*.jsonl found under: {args.cache_dir}")

    if args.max_step_files > 0:
        step_files = step_files[-args.max_step_files :]

    # sample_id -> list[(step, response_str)]
    per_sample_candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for step, path in step_files:
        per_step_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("sample_id"))
                if sid is None or sid == "None":
                    continue
                per_step_by_sample[sid].append(rec)

        # pick one response per sample in this step file
        for sid, recs in per_step_by_sample.items():
            chosen: Optional[dict[str, Any]] = None
            if args.per_step_pick == "first":
                chosen = recs[0]
            elif args.per_step_pick == "random":
                chosen = rng.choice(recs)
            elif args.per_step_pick == "best_accuracy":
                chosen = max(recs, key=lambda r: _safe_float(r.get("accuracy", 0.0)))
            elif args.per_step_pick == "best_reward":
                chosen = max(recs, key=lambda r: _safe_float(r.get("reward", 0.0)))

            if not chosen:
                continue
            resp = chosen.get("response")
            if not isinstance(resp, str) or not resp.strip():
                continue
            per_sample_candidates[sid].append((step, resp))

    # Load base dataset and write output
    base = _load_base_dataset(dataset_path, split=split)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    written = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(base):
            if args.id_key in row and row[args.id_key] is not None:
                sid = str(row[args.id_key])
            else:
                sid = str(idx)

            cand = per_sample_candidates.get(sid, [])
            # Sort by step ascending (temporal trajectory)
            cand = sorted(cand, key=lambda x: x[0])
            solutions = [resp for _, resp in cand]

            # Deduplicate while preserving order
            seen = set()
            uniq_solutions = []
            for s in solutions:
                key = s.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                uniq_solutions.append(s)
            solutions = uniq_solutions

            if args.max_candidates > 0 and len(solutions) > args.max_candidates:
                # keep a diverse subset: sample uniformly across the list
                if len(solutions) == 0:
                    solutions = []
                else:
                    # deterministic subset given seed
                    idxs = list(range(len(solutions)))
                    rng.shuffle(idxs)
                    idxs = sorted(idxs[: args.max_candidates])
                    solutions = [solutions[i] for i in idxs]

            if (not solutions) and (not args.keep_empty):
                skipped += 1
                continue

            prompt = row.get(args.prompt_key)
            answer = row.get(args.answer_key)
            if prompt is None or answer is None:
                raise KeyError(
                    f"Row missing prompt/answer keys. prompt_key={args.prompt_key}, answer_key={args.answer_key}, "
                    f"available_keys={list(row.keys())}"
                )

            out_row = {
                args.prompt_key: prompt,
                args.answer_key: answer,
                "solutions": solutions,
                "sample_id": sid,
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "out": os.path.abspath(args.out),
                "written": written,
                "skipped_no_candidates": skipped,
                "num_step_files_used": len(step_files),
                "step_range_used": [step_files[0][0], step_files[-1][0]] if step_files else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


