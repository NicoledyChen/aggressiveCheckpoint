#!/usr/bin/env python3
"""
Self-consistency evaluation on HuggingFaceH4/MATH-500 (sample 50) with an OpenAI-compatible API (e.g. DeepInfra).

What it does:
- Downloads HF datasets: math-ai/aime24 and HuggingFaceH4/MATH-500
- Samples N=50 from MATH-500
- Runs temp=0 once => pass@1 (accuracy)
- Runs temp=0.7 with total K=256 samples (in request order) => pass@k for k in [8..256]
- Writes all raw dataset rows + raw API responses to JSON (JSONL) and updates progress for a live HTML dashboard.

Env vars:
- OPENAI_API_KEY or DEEPINFRA_API_KEY
- OPENAI_BASE_URL (default: https://api.deepinfra.com/v1/openai)

Example:
  python eval_self_consistency.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as _dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset
import openai
from openai import AsyncOpenAI

try:
    import tiktoken
except Exception as e:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    # EasyR1 uses mathruler for robust math answer grading (e.g. 11/2 == 5.5).
    from mathruler.grader import grade_answer  # type: ignore
except Exception:  # pragma: no cover
    grade_answer = None  # type: ignore


DEFAULT_AIME_DATASET = "math-ai/aime24"
DEFAULT_MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"

UTC = getattr(_dt, "UTC", _dt.timezone.utc)


def utc_now_iso() -> str:
    return _dt.datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_filename_component(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:80] if len(s) > 80 else s


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _get_encoding(tokenizer_hint: Optional[str]) -> Optional[Any]:
    if tiktoken is None:
        return None
    if tokenizer_hint:
        try:
            return tiktoken.encoding_for_model(tokenizer_hint)
        except Exception:
            pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str, encoding: Optional[Any]) -> int:
    if not encoding:
        # Rough fallback: ~4 chars per token for English-ish text.
        return max(1, len(text) // 4)
    return len(encoding.encode(text))


def estimate_chat_tokens(messages: Sequence[Dict[str, str]], encoding: Optional[Any]) -> int:
    # Lightweight approximation: count the concatenated content plus role markers.
    # Avoid model-specific chat token accounting; we only need a safe upper bound.
    joined = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        joined += f"<|{role}|>\n{content}\n"
    return estimate_tokens(joined, encoding)


FINAL_RE = re.compile(r"(?is)\bFINAL\s*:\s*(.+?)\s*$")
BOXED_RE = re.compile(r"(?is)\\boxed\{([^}]*)\}")
HASH_RE = re.compile(r"(?m)^\s*####\s*(.+?)\s*$")


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    s = text.strip()

    m = FINAL_RE.search(s)
    if m:
        return m.group(1).strip()

    boxed = BOXED_RE.findall(s)
    if boxed:
        return boxed[-1].strip()

    hm = HASH_RE.findall(s)
    if hm:
        return hm[-1].strip()

    # Fallback: take last non-empty line.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s


def _strip_latex_wrappers(s: str) -> str:
    s = s.replace("$", "")
    s = re.sub(r"\\(displaystyle|left|right)\b", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    return s


def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    s = _strip_latex_wrappers(s)
    # Normalize common latex fractions.
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # Remove whitespace and surrounding punctuation.
    s = re.sub(r"\s+", "", s)
    s = s.strip().strip(".")
    # Unwrap redundant braces.
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return s


def try_parse_rational(s: str) -> Optional[Tuple[int, int]]:
    s = s.strip()
    # Support formats like (a)/(b) or a/b
    m = re.fullmatch(r"\(?(-?\d+)\)?/\(?(-?\d+)\)?", s)
    if not m:
        return None
    num = int(m.group(1))
    den = int(m.group(2))
    if den == 0:
        return None
    # Normalize sign
    if den < 0:
        num, den = -num, -den
    g = abs(_gcd(num, den))
    return (num // g, den // g)


def try_parse_decimal_as_rational(s: str) -> Optional[Tuple[int, int]]:
    """
    Convert a terminating decimal like -12.3400 into a reduced rational (num, den).
    """
    s = s.strip()
    m = re.fullmatch(r"(-?\d+)\.(\d+)", s)
    if not m:
        return None
    ip = int(m.group(1))
    fp = m.group(2)
    if fp == "":
        return (ip, 1)
    den = 10 ** len(fp)
    num = abs(ip) * den + int(fp)
    if ip < 0:
        num = -num
    g = abs(_gcd(num, den))
    return (num // g, den // g)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def answers_equal(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    # Integer compare
    if re.fullmatch(r"-?\d+", p) and re.fullmatch(r"-?\d+", g):
        return int(p) == int(g)

    # Rational compare (including decimal<->fraction equivalence)
    pr = try_parse_rational(p) or try_parse_decimal_as_rational(p)
    gr = try_parse_rational(g) or try_parse_decimal_as_rational(g)
    if pr and gr:
        return pr == gr

    # Float compare (very cautious)
    if re.fullmatch(r"-?\d+(\.\d+)?", p) and re.fullmatch(r"-?\d+(\.\d+)?", g):
        try:
            return abs(float(p) - float(g)) <= 1e-9
        except Exception:
            pass

    return False


def is_correct(pred: str, gold: str) -> bool:
    """
    Prefer EasyR1's mathruler grading when available; fall back to lightweight equivalence.
    """
    if grade_answer is not None:
        try:
            return bool(grade_answer(pred, gold))
        except Exception:
            pass
    return answers_equal(pred, gold)


def safe_model_dump(obj: Any) -> Any:
    # openai>=1.* returns pydantic models.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


async def with_retry(
    fn,
    *,
    retries: int,
    base_delay_s: float,
    max_delay_s: float,
    retry_on: Tuple[type, ...],
):
    attempt = 0
    while True:
        try:
            return await fn()
        except retry_on as e:
            # Never retry invalid-request style 4xx errors (except 429 / RateLimitError).
            if isinstance(e, openai.UnprocessableEntityError):
                raise
            if isinstance(e, openai.APIStatusError):
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and not isinstance(e, openai.RateLimitError):
                    raise

            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            # Add a little jitter.
            delay *= 0.8 + random.random() * 0.4
            print(f"[retry] attempt={attempt}/{retries} sleeping={delay:.2f}s err={type(e).__name__}: {e}", file=sys.stderr)
            await asyncio.sleep(delay)


def parse_k_values(s: str) -> List[int]:
    """
    Accepts:
    - "8,16,32"
    - "8-256" (inclusive)
    - "8..256" (inclusive)
    - mixed: "8-16,32,64"
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        m = re.fullmatch(r"(\d+)\s*(?:-|\.\.)\s*(\d+)", p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(b, a + 1))
            continue
        out.append(int(p))
    out = sorted(set(k for k in out if k > 0))
    return out


def build_messages(problem: str) -> List[Dict[str, str]]:
    # Keep outputs short to reduce JSON size while keeping accuracy reasonable.
    system = (
        "You are a careful math contest solver. "
        "Think through the solution internally, but do NOT show your reasoning. "
        "Return ONLY the final answer on a single line in the form:\n"
        "FINAL: <answer>"
    )
    user = f"Problem:\n{problem}\n\nReturn only the final line."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compute_pass_at_k(correct_flags: Sequence[bool], k_values: Sequence[int]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for k in k_values:
        kk = min(k, len(correct_flags))
        out[f"pass@{k}"] = any(correct_flags[:kk]) if kk > 0 else False
    return out


def update_aggregate_metrics(
    completed_records: Sequence[Dict[str, Any]], k_values: Sequence[int]
) -> Dict[str, Any]:
    n = len(completed_records)
    if n == 0:
        return {"n": 0, "pass@1": None, "pass@k": {f"pass@{k}": None for k in k_values}}

    pass1 = sum(1 for r in completed_records if r.get("pass1", {}).get("correct") is True) / n
    passk: Dict[str, float] = {}
    for k in k_values:
        key = f"pass@{k}"
        passk[key] = sum(1 for r in completed_records if r.get("passk", {}).get(key) is True) / n
    return {"n": n, "pass@1": pass1, "pass@k": passk}


def extract_max_n_from_422(err: openai.UnprocessableEntityError) -> Optional[int]:
    """
    Providers like DeepInfra may reject `n` with a body like:
      {'detail': [{'loc': ['body','n'], 'ctx': {'le': 4}, ...}]}
    """
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if not isinstance(detail, list):
        return None
    for d in detail:
        if not isinstance(d, dict):
            continue
        loc = d.get("loc")
        if isinstance(loc, list) and loc and loc[-1] == "n":
            ctx = d.get("ctx")
            if isinstance(ctx, dict) and "le" in ctx:
                try:
                    return int(ctx["le"])
                except Exception:
                    return None
    return None


async def main_async(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        print("Missing API key. Set --api-key or env OPENAI_API_KEY/DEEPINFRA_API_KEY.", file=sys.stderr)
        return 2

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL

    run_id = args.run_id or f"{_dt.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{sanitize_filename_component(args.model)}"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Download datasets (aime24 is downloaded but not used for sampling here).
    print(f"[dataset] downloading {args.aime_dataset} ...")
    _ = load_dataset(args.aime_dataset)
    print(f"[dataset] downloading {args.math_dataset} ...")
    math_ds_dict = load_dataset(args.math_dataset)
    if args.math_split and args.math_split in math_ds_dict:
        math_ds = math_ds_dict[args.math_split]
        used_split = args.math_split
    else:
        # Prefer test split if available.
        used_split = "test" if "test" in math_ds_dict else next(iter(math_ds_dict.keys()))
        math_ds = math_ds_dict[used_split]

    rng = random.Random(args.seed)
    total = min(args.sample_size, len(math_ds))
    indices = rng.sample(range(len(math_ds)), total)
    sampled = math_ds.select(indices)

    encoding = _get_encoding(args.tokenizer_hint or args.model)

    config = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "model": args.model,
        "base_url": base_url,
        "datasets": {"aime24": args.aime_dataset, "math500": args.math_dataset, "math_split": used_split},
        "sample_size": total,
        "sample_indices": indices,
        "seed": args.seed,
        "max_context_tokens": args.max_context_tokens,
        "token_margin": args.token_margin,
        "max_new_tokens_cap": args.max_new_tokens_cap,
        "pass1_temperature": 0.0,
        "sc_temperature": args.sc_temperature,
        "sc_total": args.sc_total,
        "n_per_request": args.n_per_request,
        "k_values": args.k_values,
        "prompt_format": "system+user with FINAL: prefix",
    }
    atomic_write_json(run_dir / "config.json", config)

    # Save sampled raw rows.
    atomic_write_json(run_dir / "samples.json", [dict(r) for r in sampled])

    # Copy a run-local dashboard for convenience.
    dashboard_src = Path(__file__).with_name("dashboard.html")
    if dashboard_src.exists():
        (run_dir / "dashboard.html").write_text(dashboard_src.read_text(encoding="utf-8"), encoding="utf-8")

    results_path = run_dir / "results.jsonl"
    requests_path = run_dir / "requests.jsonl"
    progress_path = run_dir / "progress.json"
    summary_path = run_dir / "summary.json"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    completed_records: List[Dict[str, Any]] = []

    progress_state: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "total": total,
        "completed": 0,
        "current": None,
        "aggregate": update_aggregate_metrics(completed_records, args.k_values),
        "errors": 0,
    }
    atomic_write_json(progress_path, progress_state)

    async def chat_create(payload: Dict[str, Any]) -> Any:
        async def _call():
            return await client.chat.completions.create(**payload)

        # Most transient errors surface as Exception; keep retry broad but bounded.
        return await with_retry(
            _call,
            retries=args.retries,
            base_delay_s=args.retry_base_delay_s,
            max_delay_s=args.retry_max_delay_s,
            retry_on=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIStatusError,
            ),
        )

    for i, row in enumerate(sampled):
        sample_dict = dict(row)
        problem = sample_dict.get(args.problem_key, "")
        gold = sample_dict.get(args.answer_key, "")

        sample_id = sample_dict.get("unique_id") or sample_dict.get("id") or str(i)

        messages = build_messages(problem)
        prompt_tokens = estimate_chat_tokens(messages, encoding)
        max_tokens = min(args.max_new_tokens_cap, max(1, args.max_context_tokens - prompt_tokens - args.token_margin))

        record: Dict[str, Any] = {
            "sample_index": i,
            "dataset_index": indices[i],
            "sample_id": sample_id,
            "sample": sample_dict,
            "prompt": {"messages": messages, "prompt_tokens_est": prompt_tokens, "max_tokens": max_tokens},
            "gold": {"answer": gold, "normalized": normalize_answer(gold)},
            "pass1": None,
            "sc": {"temperature": args.sc_temperature, "total": args.sc_total, "requests": [], "generations": []},
            "passk": None,
            "timing": {"started_at": utc_now_iso(), "ended_at": None},
            "error": None,
        }

        progress_state["current"] = {
            "sample_index": i,
            "sample_id": sample_id,
            "dataset_index": indices[i],
            "phase": "pass@1",
            "sc_done": 0,
            "sc_total": args.sc_total,
        }
        progress_state["updated_at"] = utc_now_iso()
        atomic_write_json(progress_path, progress_state)

        try:
            # pass@1 (temp=0)
            t0 = time.time()
            payload1 = {
                "model": args.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "n": 1,
            }
            resp1 = await chat_create(payload1)
            dt1 = time.time() - t0
            text1 = (resp1.choices[0].message.content or "") if getattr(resp1, "choices", None) else ""
            pred1 = extract_final_answer(text1)
            correct1 = is_correct(pred1, gold)
            req_id_1 = f"{sample_id}::pass1"
            append_jsonl(
                requests_path,
                {
                    "request_id": req_id_1,
                    "sample_index": i,
                    "sample_id": sample_id,
                    "kind": "pass1",
                    "request": payload1,
                    "raw_response": safe_model_dump(resp1),
                    "latency_s": dt1,
                    "created_at": utc_now_iso(),
                },
            )
            record["pass1"] = {
                "request_id": req_id_1,
                "request": payload1,
                "text": text1,
                "pred": pred1,
                "pred_normalized": normalize_answer(pred1),
                "correct": bool(correct1),
                "latency_s": dt1,
            }

            # Self-consistency: accumulate generations to length sc_total in request order.
            remaining = args.sc_total
            gen_idx = 0
            n_limit = max(1, int(args.n_per_request))
            chunk_idx = 0
            while remaining > 0:
                n_chunk = min(n_limit, remaining)
                progress_state["current"]["phase"] = "pass@k"
                progress_state["current"]["sc_done"] = args.sc_total - remaining
                progress_state["updated_at"] = utc_now_iso()
                atomic_write_json(progress_path, progress_state)

                t1 = time.time()
                payload_sc = {
                    "model": args.model,
                    "messages": messages,
                    "temperature": args.sc_temperature,
                    "max_tokens": max_tokens,
                    "n": n_chunk,
                }
                try:
                    resp_sc = await chat_create(payload_sc)
                except openai.UnprocessableEntityError as e:
                    # Auto-adjust provider max-n (DeepInfra often limits n<=4).
                    max_n = extract_max_n_from_422(e)
                    if max_n is not None and n_chunk > max_n and max_n >= 1:
                        n_limit = max_n
                        record.setdefault("provider_limits", {})["max_n"] = max_n
                        continue
                    raise
                dt_sc = time.time() - t1

                req_id_sc = f"{sample_id}::sc::{chunk_idx}"
                append_jsonl(
                    requests_path,
                    {
                        "request_id": req_id_sc,
                        "sample_index": i,
                        "sample_id": sample_id,
                        "kind": "sc",
                        "chunk_index": chunk_idx,
                        "request": payload_sc,
                        "raw_response": safe_model_dump(resp_sc),
                        "latency_s": dt_sc,
                        "created_at": utc_now_iso(),
                    },
                )
                record["sc"]["requests"].append(
                    {"request_id": req_id_sc, "request": payload_sc, "latency_s": dt_sc}
                )

                # Extract in order of returned choices.
                choices = getattr(resp_sc, "choices", []) or []
                for ch in choices:
                    content = (ch.message.content or "") if getattr(ch, "message", None) else ""
                    pred = extract_final_answer(content)
                    ok = is_correct(pred, gold)
                    record["sc"]["generations"].append(
                        {
                            "index": gen_idx,
                            "text": content,
                            "pred": pred,
                            "pred_normalized": normalize_answer(pred),
                            "correct": bool(ok),
                        }
                    )
                    gen_idx += 1

                remaining -= n_chunk
                chunk_idx += 1

            # Compute pass@k.
            correct_flags = [g["correct"] for g in record["sc"]["generations"]]
            record["passk"] = compute_pass_at_k(correct_flags, args.k_values)

            record["timing"]["ended_at"] = utc_now_iso()
            append_jsonl(results_path, record)
            completed_records.append(record)

            progress_state["completed"] = len(completed_records)
            progress_state["aggregate"] = update_aggregate_metrics(completed_records, args.k_values)
            progress_state["current"] = None
            progress_state["updated_at"] = utc_now_iso()
            atomic_write_json(progress_path, progress_state)

        except Exception as e:
            record["error"] = {"type": type(e).__name__, "message": str(e)}
            record["timing"]["ended_at"] = utc_now_iso()
            append_jsonl(results_path, record)

            progress_state["errors"] += 1
            progress_state["current"] = None
            progress_state["updated_at"] = utc_now_iso()
            atomic_write_json(progress_path, progress_state)
            continue

    summary = {
        "run_id": run_id,
        "finished_at": utc_now_iso(),
        "aggregate": update_aggregate_metrics(completed_records, args.k_values),
        "errors": progress_state["errors"],
        "total": total,
        "completed": len(completed_records),
    }
    atomic_write_json(summary_path, summary)

    print(f"[done] run_dir={run_dir}")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="OpenAI-compatible model name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    p.add_argument("--api-key", default=None, help="API key (or use env OPENAI_API_KEY / DEEPINFRA_API_KEY)")
    p.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    p.add_argument("--run-id", default=None, help="Optional: override run directory name under --runs-dir")

    p.add_argument("--aime-dataset", default=DEFAULT_AIME_DATASET)
    p.add_argument("--math-dataset", default=DEFAULT_MATH500_DATASET)
    p.add_argument("--math-split", default=None, help="Split name for MATH-500; if omitted, prefers 'test' if present.")

    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")

    p.add_argument("--problem-key", default="problem")
    p.add_argument("--answer-key", default="answer")

    p.add_argument("--max-context-tokens", type=int, default=4096)
    p.add_argument("--token-margin", type=int, default=64, help="Safety margin subtracted from context window.")
    p.add_argument("--max-new-tokens-cap", type=int, default=1024, help="Cap for max_tokens; still clipped by context budget.")
    p.add_argument("--tokenizer-hint", default=None, help="Optional tiktoken model name for better token estimation.")

    p.add_argument("--sc-temperature", type=float, default=0.7)
    p.add_argument("--sc-total", type=int, default=256)
    p.add_argument(
        "--n-per-request",
        type=int,
        default=4,
        help="Number of completions per API call (chunk size). DeepInfra often limits n<=4.",
    )
    p.add_argument("--k-values", type=parse_k_values, default=parse_k_values("8-256"))

    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--retry-base-delay-s", type=float, default=1.0)
    p.add_argument("--retry-max-delay-s", type=float, default=20.0)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


