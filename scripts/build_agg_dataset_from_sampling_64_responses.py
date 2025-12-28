#!/usr/bin/env python3
"""
Build an EasyR1 aggregation dataset (problem/answer/solutions[]) from temporal-forgetting style
folders.

Expected layout (example):
  sampling_64_responses/
    Qwen__Qwen2.5-7B/
      model_final_answer_AIME.json
      model_final_answer_AIME25.json
      model_final_answer_AMC.json
      samples_AIME_*.jsonl
      samples_AIME25_*.jsonl
      samples_AMC_*.jsonl
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

def _find_samples_jsonl(ckpt_dir: str, dataset_name: str) -> Optional[str]:
    prefix = f"samples_{dataset_name}_"
    candidates = [
        os.path.join(ckpt_dir, n)
        for n in os.listdir(ckpt_dir)
        if n.startswith(prefix) and n.endswith(".jsonl")
    ]
    # Prefer the latest by name (timestamps in filename)
    candidates.sort()
    return candidates[-1] if candidates else None


def _load_samples_full_responses(ckpt_dir: str, dataset_name: str) -> Optional[dict[int, dict[str, Any]]]:
    """
    Parse `samples_{DATASET}_*.jsonl` produced by lm-eval style runners.
    Each line stores a single question and includes:
      - doc_id (int): question index
      - doc.problem / doc.answer
      - resps[0]: list[str] of sampled full responses (often length 64)
    """
    path = _find_samples_jsonl(ckpt_dir, dataset_name)
    if path is None or (not os.path.exists(path)):
        return None
    by_index: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("doc_id", None)
            if idx is None:
                continue
            idx = int(idx)
            doc = obj.get("doc") or {}
            problem = _coerce_str(doc.get("problem") or doc.get("question") or "").strip()
            answer = _coerce_str(doc.get("answer") or "").strip()
            resps = obj.get("resps") or []
            # schema: resps = [ [resp0, resp1, ...] ]
            if resps and isinstance(resps[0], list):
                full_resps = [str(r) for r in resps[0] if isinstance(r, (str, int, float))]
            else:
                full_resps = []
            by_index[idx] = {"problem": problem, "answer": answer, "full_responses": full_resps}
    return by_index


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

def _truncate_text(text: str, max_chars: Optional[int], mode: str) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if mode == "head":
        return text[:max_chars].rstrip() + "\n...[truncated]..."
    if mode == "tail":
        return "...[truncated]...\n" + text[-max_chars:].lstrip()
    if mode == "head_tail":
        head = max_chars // 2
        tail = max_chars - head
        return (
            text[:head].rstrip()
            + "\n...[truncated]...\n"
            + text[-tail:].lstrip()
        )
    if mode == "none":
        return text
    raise ValueError(f"Unknown response_truncate mode: {mode}")


def _pick_texts(texts: list[str], k: int, pick: str, rng: random.Random, unique: bool) -> list[str]:
    cleaned: list[str] = []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        cleaned.append(t)
    if unique:
        seen = set()
        uniq = []
        for t in cleaned:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        cleaned = uniq
    if k <= 0:
        return []
    if k >= len(cleaned):
        out = list(cleaned)
        if pick == "random":
            rng.shuffle(out)
        return out
    if pick == "first":
        return cleaned[:k]
    if pick == "random":
        return rng.sample(cleaned, k)
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
    ap.add_argument(
        "--solutions_source",
        choices=["final_answer", "full_response", "full_response_with_extracted_answer"],
        default="final_answer",
        help=(
            "final_answer: build candidates from model_final_answer_* extracted resp_answer (short). "
            "full_response: build candidates from samples_* full generations. "
            "full_response_with_extracted_answer: prepend extracted Answer: to each full response."
        ),
    )
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
    ap.add_argument(
        "--response_max_chars",
        type=int,
        default=2000,
        help="When solutions_source includes full_response, truncate each response to this many chars (0 disables).",
    )
    ap.add_argument(
        "--response_truncate",
        choices=["head", "tail", "head_tail", "none"],
        default="head_tail",
        help="How to truncate full responses when response_max_chars is set.",
    )
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
    # We keep a per-index dict with at least: problem, answer, and candidate sources.
    ckpt_data: list[tuple[Ckpt, dict[int, dict[str, Any]]]] = []
    for ckpt in ckpts:
        by_index: dict[int, dict[str, Any]] = {}
        if args.solutions_source == "final_answer":
            data = _load_model_final_answer(ckpt.path, args.dataset)
            if data is None:
                continue
            for row in data:
                idx = row.get("index")
                if idx is None:
                    continue
                by_index[int(idx)] = row
        elif args.solutions_source in ("full_response", "full_response_with_extracted_answer"):
            samples = _load_samples_full_responses(ckpt.path, args.dataset)
            if samples is None:
                continue
            if args.solutions_source == "full_response_with_extracted_answer":
                finals = _load_model_final_answer(ckpt.path, args.dataset)
                if finals is None:
                    continue
                finals_by_index: dict[int, dict[str, Any]] = {int(r["index"]): r for r in finals if "index" in r}
                # join
                common_idx = set(samples.keys()) & set(finals_by_index.keys())
                for idx in common_idx:
                    srow = samples[idx]
                    frow = finals_by_index[idx]
                    # build resp_idx -> resp_answer map
                    ans_map = {}
                    for rr in (frow.get("responses") or []):
                        ridx = rr.get("resp_idx")
                        if ridx is None:
                            continue
                        ans = _coerce_str(rr.get("resp_answer", "")).strip()
                        # Filter placeholders / empty
                        if not ans or ("{{" in ans and "}}" in ans):
                            continue
                        ans_map[int(ridx)] = ans
                    full_resps: list[str] = srow.get("full_responses") or []
                    merged: list[str] = []
                    for ridx, text in enumerate(full_resps):
                        text = _coerce_str(text).strip()
                        if not text:
                            continue
                        ans = ans_map.get(ridx, "")
                        prefix = f"Answer: {ans}\n" if ans else ""
                        merged.append(prefix + text)
                    by_index[idx] = {
                        "problem": srow.get("problem", ""),
                        "answer": srow.get("answer", ""),
                        "candidates": merged,
                    }
            else:
                for idx, srow in samples.items():
                    by_index[idx] = {
                        "problem": srow.get("problem", ""),
                        "answer": srow.get("answer", ""),
                        "candidates": srow.get("full_responses") or [],
                    }
        else:
            raise ValueError(f"Unsupported solutions_source: {args.solutions_source}")

        if by_index:
            ckpt_data.append((ckpt, by_index))

    if not ckpt_data:
        raise RuntimeError("No usable checkpoint data found. Check --solutions_source and input folder contents.")

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
        if args.solutions_source in ("full_response", "full_response_with_extracted_answer"):
            # joined samples path stores problem/answer under these keys
            problem = _coerce_str(base.get("problem", problem)).strip()
            answer = _coerce_str(base.get("answer", answer)).strip()
        if not problem or answer == "":
            continue

        solutions: list[str] = []
        for ckpt, m in ckpt_data:
            item = m[idx]
            if args.solutions_source == "final_answer":
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
            else:
                cand_list = item.get("candidates") or []
                picked = _pick_texts(
                    texts=[_coerce_str(x) for x in cand_list],
                    k=args.k_per_ckpt,
                    pick=args.pick,
                    rng=rng,
                    unique=args.unique_within_ckpt,
                )
                for t in picked:
                    t = _truncate_text(t, max_chars=args.response_max_chars, mode=args.response_truncate)
                    if args.include_ckpt_name:
                        solutions.append(f"[{ckpt.name}]\n{t}")
                    else:
                        solutions.append(t)

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



