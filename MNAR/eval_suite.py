#!/usr/bin/env python3
"""
Evaluate BOTH AIME24 and MATH-500 for a given model (OpenAI-compatible endpoint).

This is a thin wrapper that runs `eval_self_consistency.py` twice into a shared parent folder:
- <runs_dir>/<suite_id>/aime24/
- <runs_dir>/<suite_id>/math500/

Each sub-run gets its own dashboard and per-sample JSONL, but you also get a suite_summary.json
for training-dynamics sweeps.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import eval_self_consistency as esc


DEFAULT_MATH500 = "HuggingFaceH4/MATH-500"
DEFAULT_AIME24 = "math-ai/aime24"


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def run_one(run_dir: Path, args_list: list[str]) -> Dict[str, Any]:
    parser = esc.build_arg_parser()
    parsed = parser.parse_args(args_list)
    rc = await esc.main_async(parsed)
    if rc != 0:
        raise RuntimeError(f"eval_self_consistency failed: rc={rc} args={args_list}")
    summary_path = run_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="OpenAI-compatible model name OR deploy_id:xxxx")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)

    p.add_argument("--suite-id", required=True, help="Folder name under --runs-dir for this suite run")
    p.add_argument("--runs-dir", default="runs_suite")

    p.add_argument("--math-dataset", default=DEFAULT_MATH500)
    p.add_argument("--aime-dataset", default=DEFAULT_AIME24)

    p.add_argument("--math-sample-size", type=int, default=50)
    p.add_argument("--aime-sample-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-context-tokens", type=int, default=4096)
    p.add_argument("--token-margin", type=int, default=64)
    p.add_argument("--max-new-tokens-cap", type=int, default=1024)
    p.add_argument("--tokenizer-hint", default=None)

    p.add_argument("--sc-temperature", type=float, default=0.7)
    p.add_argument("--sc-total", type=int, default=256)
    p.add_argument("--n-per-request", type=int, default=4)

    p.add_argument("--k-values", default="8-256", help="e.g. 8-256 or 8,16,32")
    return p


async def main_async(args: argparse.Namespace) -> int:
    suite_root = Path(args.runs_dir) / args.suite_id
    suite_root.mkdir(parents=True, exist_ok=True)

    common = []
    if args.api_key:
        common += ["--api-key", args.api_key]
    if args.base_url:
        common += ["--base-url", args.base_url]

    common += [
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--token-margin",
        str(args.token_margin),
        "--max-new-tokens-cap",
        str(args.max_new_tokens_cap),
        "--sc-temperature",
        str(args.sc_temperature),
        "--sc-total",
        str(args.sc_total),
        "--n-per-request",
        str(args.n_per_request),
        "--k-values",
        str(args.k_values),
    ]
    if args.tokenizer_hint:
        common += ["--tokenizer-hint", args.tokenizer_hint]

    # AIME24 (answer key is 'solution' in math-ai/aime24)
    math_dir = suite_root / "math500"
    math_run_id = "math500"
    math_args = (
        common
        + [
            "--runs-dir",
            str(suite_root),
            "--run-id",
            str(math_run_id),
            "--aime-dataset",
            args.aime_dataset,  # just to satisfy download contract
            "--math-dataset",
            args.math_dataset,
            "--sample-size",
            str(args.math_sample_size),
            "--problem-key",
            "problem",
            "--answer-key",
            "answer",
        ]
    )
    aime_dir = suite_root / "aime24"
    aime_run_id = "aime24"
    aime_args = (
        common
        + [
            "--runs-dir",
            str(suite_root),
            "--run-id",
            str(aime_run_id),
            "--aime-dataset",
            args.aime_dataset,
            "--math-dataset",
            args.aime_dataset,
            "--sample-size",
            str(args.aime_sample_size),
            "--problem-key",
            "problem",
            "--answer-key",
            "solution",
        ]
    )

    # Run AIME24 first for faster early signal in sweeps.
    aime_summary = await run_one(aime_dir, aime_args)
    math_summary = await run_one(math_dir, math_args)

    suite_summary = {
        "suite_id": args.suite_id,
        "model": args.model,
        "datasets": {"math500": math_summary, "aime24": aime_summary},
    }
    atomic_write_json(suite_root / "suite_summary.json", suite_summary)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


