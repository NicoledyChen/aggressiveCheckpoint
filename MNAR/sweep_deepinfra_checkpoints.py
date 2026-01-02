#!/usr/bin/env python3
"""
Checkpoint sweep:
  - Deploy HF checkpoints on DeepInfra (serverless/custom LLM)
  - Evaluate AIME24 + MATH-500
  - Delete deployments afterwards
  - Cap maximum concurrent active deployments to N

Outputs:
  sweeps/<sweep_id>/
    - sweep_config.json
    - sweep_results.jsonl    (one line per checkpoint)
    - sweep_results.csv      (flat summary)
    - suites/<step_xx>/...   (per-dataset run folders + dashboards)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepinfra_deploy import DeepInfraDeployer, DeepInfraDeployError, default_gpu_preference

import eval_suite


UTC = getattr(_dt, "UTC", _dt.timezone.utc)


def utc_now_iso() -> str:
    return _dt.datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def summarize_existing_results(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Deduplicate by step: keep the last seen row per step.
    """
    step_map: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        try:
            step = int(r.get("step"))
        except Exception:
            continue
        step_map[step] = r
    return step_map


def load_suite_summary_if_exists(out_root: Path, step: int) -> Optional[Dict[str, Any]]:
    suite_id = f"step_{step:02d}"
    p = out_root / "suites" / suite_id / "suite_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def sanitize_component(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:120] if len(s) > 120 else s


def flatten_metrics(prefix: str, metrics: Dict[str, Any], out: Dict[str, Any]) -> None:
    # metrics: {"n":..., "pass@1":..., "pass@k":{...}, "sc@k":{...}}
    out[f"{prefix}.n"] = metrics.get("n")
    out[f"{prefix}.pass@1"] = metrics.get("pass@1")
    for k, v in (metrics.get("pass@k") or {}).items():
        out[f"{prefix}.{k}"] = v
    for k, v in (metrics.get("sc@k") or {}).items():
        out[f"{prefix}.{k}"] = v


def _get_str(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v)
        if s:
            return s
    return default


def _extract_deploy_id(d: Dict[str, Any]) -> str:
    return _get_str(d, ["deploy_id", "deployId", "id"], "")


def _extract_model_name(d: Dict[str, Any]) -> str:
    return _get_str(d, ["model_name", "modelName", "name"], "")


def _extract_status(d: Dict[str, Any]) -> str:
    return _get_str(d, ["status", "state"], "")


def _status_is_reusable(status: str) -> bool:
    s = (status or "").strip().lower()
    if not s:
        # Unknown → allow (we'll confirm with GET /deploy/{id}).
        return True
    return s not in {"deleted", "deleting", "failed", "error"}


async def eval_one_checkpoint(
    *,
    step: int,
    hf_repo: str,
    deploy_name: str,
    gpu: Optional[str],
    num_gpus: int,
    max_batch_size: int,
    min_instances: int,
    max_instances: int,
    hf_token: Optional[str],
    base_openai_url: Optional[str],
    api_key: str,
    out_root: Path,
    max_context_tokens: int,
    sc_total: int,
    n_per_request: int,
    math_sample_size: int,
    aime_sample_size: int,
    seed: int,
    tokenizer_hint: Optional[str],
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with sem:
        suite_id = f"step_{step:02d}"
        suites_dir = out_root / "suites"
        suite_root = suites_dir / suite_id
        suite_root.mkdir(parents=True, exist_ok=True)

        # Deploy → eval → delete (always).
        deploy_id = None
        deploy_status = None
        t0 = utc_now_iso()
        deletion_error: Optional[str] = None
        reused_existing: bool = False
        res: Dict[str, Any] = {}

        async with DeepInfraDeployer(api_key=api_key) as di:
            chosen_gpu = gpu
            if not chosen_gpu:
                chosen_gpu = await di.choose_gpu(preferred=default_gpu_preference(), num_gpus=num_gpus)

            try:
                # --- Avoid duplicate creation: reuse existing deployment with the same model_name ---
                try:
                    matches: List[Dict[str, Any]] = []
                    for d in await di.list_deployments():
                        if _extract_model_name(d) == deploy_name:
                            matches.append(d)

                    # Pick the first reusable match (skip deleted/deleting/failed/error).
                    picked: Optional[Dict[str, Any]] = None
                    for d in matches:
                        if not _status_is_reusable(_extract_status(d)):
                            continue
                        if _extract_deploy_id(d):
                            picked = d
                            break

                    if picked is None and matches:
                        # Helpful log: name exists but all matches are non-reusable (e.g. deleted).
                        st = _extract_status(matches[0])
                        print(
                            f"[step {step:02d}] found existing deployment but status={st!r} (non-reusable); will deploy new"
                        )

                    if picked:
                        cand_id = _extract_deploy_id(picked)
                        cand_status = _extract_status(picked)
                        if cand_id:
                            # Confirm with GET /deploy/{id} in case list is stale.
                            try:
                                d_full = await di.get_deployment(cand_id)
                                st_full = (d_full.status or "").strip()
                                if _status_is_reusable(st_full):
                                    deploy_id = cand_id
                                    deploy_status = st_full
                                    reused_existing = True
                                    print(
                                        f"[step {step:02d}] found existing deployment; reuse deploy_id={deploy_id} status={deploy_status}"
                                    )
                                else:
                                    print(
                                        f"[step {step:02d}] existing deployment non-reusable after confirm: deploy_id={cand_id} status={st_full!r}; will deploy new"
                                    )
                            except Exception as e:
                                print(
                                    f"[step {step:02d}] WARN get_deployment failed for candidate deploy_id={cand_id}; will deploy new err={type(e).__name__}: {e}"
                                )
                        else:
                            # No deploy_id in listing (unexpected); fall back to deploy.
                            print(
                                f"[step {step:02d}] WARN list_deployments match missing deploy_id (status={cand_status!r}); will deploy new"
                            )
                except Exception as e:
                    print(f"[step {step:02d}] WARN list_deployments failed; will try deploy anyway err={type(e).__name__}: {e}")

                if not deploy_id:
                    print(f"[step {step:02d}] deploy start hf_repo={hf_repo} gpu={chosen_gpu} num_gpus={num_gpus}")
                    dep = await di.deploy_llm_from_hf(
                        deploy_model_name=deploy_name,
                        hf_repo=hf_repo,
                        gpu=chosen_gpu,
                        num_gpus=num_gpus,
                        max_batch_size=max_batch_size,
                        min_instances=min_instances,
                        max_instances=max_instances,
                        hf_token=hf_token,
                    )
                    deploy_id = dep.deploy_id
                    print(f"[step {step:02d}] deploy created deploy_id={deploy_id}")

                dep2 = await di.wait_until_deployed(deploy_id)
                deploy_status = dep2.status
                print(f"[step {step:02d}] deploy ready status={deploy_status}")

                # Evaluate via OpenAI-compatible API using deploy_id:<id>
                model_for_eval = f"deploy_id:{deploy_id}"

                # Reuse eval_suite (which itself calls eval_self_consistency twice).
                suite_args = argparse.Namespace(
                    model=model_for_eval,
                    api_key=api_key,
                    base_url=base_openai_url,
                    suite_id=suite_id,
                    runs_dir=str(suites_dir),
                    math_dataset=eval_suite.DEFAULT_MATH500,
                    aime_dataset=eval_suite.DEFAULT_AIME24,
                    math_sample_size=math_sample_size,
                    aime_sample_size=aime_sample_size,
                    seed=seed,
                    max_context_tokens=max_context_tokens,
                    token_margin=64,
                    max_new_tokens_cap=1024,
                    tokenizer_hint=tokenizer_hint,
                    sc_temperature=0.7,
                    sc_total=sc_total,
                    n_per_request=n_per_request,
                    k_values="8-256",
                )
                await eval_suite.main_async(suite_args)

                suite_summary = json.loads((suite_root / "suite_summary.json").read_text(encoding="utf-8"))
                t1 = utc_now_iso()
                print(f"[step {step:02d}] eval done")

                res = {
                    "step": step,
                    "hf_repo": hf_repo,
                    "deploy_name": deploy_name,
                    "deploy_id": deploy_id,
                    "reused_existing": reused_existing,
                    "gpu": chosen_gpu,
                    "num_gpus": num_gpus,
                    "max_batch_size": max_batch_size,
                    "min_instances": min_instances,
                    "max_instances": max_instances,
                    "started_at": t0,
                    "finished_at": t1,
                    "deploy_status": deploy_status,
                    "deletion_error": deletion_error,
                    "suite": suite_summary,
                }
            except Exception as e:
                t1 = utc_now_iso()
                print(f"[step {step:02d}] ERROR {type(e).__name__}: {e}")
                res = {
                    "step": step,
                    "hf_repo": hf_repo,
                    "deploy_name": deploy_name,
                    "deploy_id": deploy_id,
                    "reused_existing": reused_existing,
                    "gpu": chosen_gpu,
                    "num_gpus": num_gpus,
                    "max_batch_size": max_batch_size,
                    "min_instances": min_instances,
                    "max_instances": max_instances,
                    "started_at": t0,
                    "finished_at": t1,
                    "deploy_status": deploy_status,
                    "deletion_error": deletion_error,
                    "error": {"type": type(e).__name__, "message": str(e)},
                }
            finally:
                if deploy_id:
                    try:
                        await di.delete(deploy_id)
                        print(f"[step {step:02d}] deploy deleted deploy_id={deploy_id}")
                    except Exception as e:
                        deletion_error = f"{type(e).__name__}: {e}"
                        print(f"[step {step:02d}] WARN delete failed deploy_id={deploy_id} err={deletion_error}")
                # Ensure deletion_error is visible in the returned row.
                if res is not None:
                    res["deletion_error"] = deletion_error

        return res


async def main_async(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPINFRA_API_KEY / OPENAI_API_KEY or --api-key")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    sweep_id = args.sweep_id or f"{_dt.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{sanitize_component(args.hf_prefix)}"
    out_root = Path(args.out_dir) / sweep_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] sweep_id={sweep_id} out_root={out_root} max_active={args.max_active}")

    cfg_path = out_root / "sweep_config.json"
    if not cfg_path.exists():
        cfg = vars(args).copy()
        cfg["sweep_id"] = sweep_id
        cfg["created_at"] = utc_now_iso()
        cfg["api_key"] = "[REDACTED]"
        if cfg.get("hf_token"):
            cfg["hf_token"] = "[REDACTED]"
        atomic_write_json(cfg_path, cfg)
    else:
        print("[sweep] resume: keeping existing sweep_config.json (not overwriting)")

    sem = asyncio.Semaphore(args.max_active)
    results_path = out_root / "sweep_results.jsonl"

    # Resolve deploy prefix (namespace) once.
    deploy_prefix = args.deploy_prefix
    if deploy_prefix and deploy_prefix.lower() == "auto":
        deploy_prefix = None
    if not deploy_prefix:
        async with DeepInfraDeployer(api_key=api_key) as di:
            deploy_prefix = await di.infer_deploy_prefix()
        if not deploy_prefix:
            raise RuntimeError(
                "Could not infer DeepInfra deployment namespace/prefix. "
                "Pass --deploy-prefix <your_deepinfra_username_or_team>."
            )
    print(f"[sweep] deploy_prefix={deploy_prefix}")

    # --- Resume support ---
    existing_rows = read_jsonl(results_path)
    step_to_row = summarize_existing_results(existing_rows)

    # Recover any missing sweep rows from existing per-step suite_summary.json (if present).
    recovered = 0
    for step in range(args.start, args.end + 1):
        if step in step_to_row:
            continue
        suite_summary = load_suite_summary_if_exists(out_root, step)
        if not suite_summary:
            continue
        hf_repo = f"{args.hf_prefix}{step}"
        repo_name = hf_repo.split("/")[-1]
        step_to_row[step] = {
            "step": step,
            "hf_repo": hf_repo,
            "deploy_name": f"{deploy_prefix}/{repo_name}",
            "deploy_id": None,
            "gpu": args.gpu,
            "num_gpus": args.num_gpus,
            "max_batch_size": args.max_batch_size,
            "min_instances": args.min_instances,
            "max_instances": args.max_instances,
            "started_at": None,
            "finished_at": None,
            "deploy_status": None,
            "deletion_error": None,
            "suite": suite_summary,
            "recovered": True,
        }
        recovered += 1

    done_steps = sorted(step_to_row.keys())
    if done_steps:
        ok_prev = sum(1 for r in step_to_row.values() if "error" not in r)
        err_prev = sum(1 for r in step_to_row.values() if "error" in r)
        print(f"[sweep] resume: found {len(done_steps)} prior results (ok={ok_prev} err={err_prev} recovered={recovered})")

    tasks = []
    steps_to_run: List[int] = []
    for step in range(args.start, args.end + 1):
        if step in step_to_row and not (args.retry_failed and "error" in step_to_row[step]):
            continue
        steps_to_run.append(step)

    print(f"[sweep] steps_to_run={len(steps_to_run)} (range {args.start}..{args.end})")
    for step in steps_to_run:
        hf_repo = f"{args.hf_prefix}{step}"
        repo_name = hf_repo.split("/")[-1]
        deploy_name = f"{deploy_prefix}/{repo_name}"
        tasks.append(
            asyncio.create_task(
                eval_one_checkpoint(
                    step=step,
                    hf_repo=hf_repo,
                    deploy_name=deploy_name,
                    gpu=args.gpu,
                    num_gpus=args.num_gpus,
                    max_batch_size=args.max_batch_size,
                    min_instances=args.min_instances,
                    max_instances=args.max_instances,
                    hf_token=hf_token,
                    base_openai_url=args.openai_base_url,
                    api_key=api_key,
                    out_root=out_root,
                    max_context_tokens=args.max_context_tokens,
                    sc_total=args.sc_total,
                    n_per_request=args.n_per_request,
                    math_sample_size=args.math_sample_size,
                    aime_sample_size=args.aime_sample_size,
                    seed=args.seed,
                    tokenizer_hint=args.tokenizer_hint,
                    sem=sem,
                )
            )
        )

    for coro in asyncio.as_completed(tasks):
        res = await coro
        append_jsonl(results_path, res)
        try:
            step_to_row[int(res["step"])] = res
        except Exception:
            pass

    # write CSV
    if step_to_row:
        csv_path = out_root / "sweep_results.csv"
        flat_rows: List[Dict[str, Any]] = []
        for step in sorted(step_to_row.keys()):
            res = step_to_row[step]
            flat: Dict[str, Any] = {
                "step": res.get("step"),
                "hf_repo": res.get("hf_repo"),
                "deploy_id": res.get("deploy_id"),
                "gpu": res.get("gpu"),
                "started_at": res.get("started_at"),
                "finished_at": res.get("finished_at"),
            }
            suite = res.get("suite") or {}
            if suite:
                for ds_name in ("math500", "aime24"):
                    ds = (suite.get("datasets") or {}).get(ds_name) or {}
                    agg = (ds.get("aggregate") or {})
                    flatten_metrics(ds_name, agg, flat)
            flat_rows.append(flat)

        cols = sorted({k for r in flat_rows for k in r.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(flat_rows, key=lambda x: x["step"]):
                w.writerow(r)

    ok_total = sum(1 for r in step_to_row.values() if "error" not in r)
    err_total = sum(1 for r in step_to_row.values() if "error" in r)
    print(f"[sweep] done ok={ok_total} err={err_total} results={results_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default="he4dBZPx7GTPxr5LLjYZSxNqLwivb2gy", help="DeepInfra API key (or env DEEPINFRA_API_KEY)")
    p.add_argument("--hf-token", default=None, help="Optional HF token for private repos (or env HF_TOKEN)")

    p.add_argument("--hf-prefix", required=True, help="HF repo prefix including owner, e.g. nicoledy/qwen2.5-math-base-limr-step-")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=41)

    p.add_argument(
        "--deploy-prefix",
        default="auto",
        help="DeepInfra namespace/prefix to create deployments under (e.g. your username/team). "
        "Use 'auto' to infer from your account's existing models/deployments.",
    )
    p.add_argument("--gpu", default="B200-180GB", help="GPU config, e.g. B200-180GB. If omitted, auto-choose (prefers B200).")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-batch-size", type=int, default=32, help="Max concurrent requests per instance")
    p.add_argument("--min-instances", type=int, default=1)
    p.add_argument("--max-instances", type=int, default=1)
    p.add_argument("--max-active", type=int, default=4, help="Max concurrent active deployments (caps total instances if max_instances=1).")

    p.add_argument("--openai-base-url", default="https://api.deepinfra.com/v1/openai")

    p.add_argument("--math-sample-size", type=int, default=50)
    p.add_argument("--aime-sample-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sc-total", type=int, default=256)
    p.add_argument("--n-per-request", type=int, default=4)
    p.add_argument("--max-context-tokens", type=int, default=4096)
    p.add_argument("--tokenizer-hint", default=None)

    p.add_argument("--out-dir", default="sweeps")
    p.add_argument("--sweep-id", default=None)
    p.add_argument("--retry-failed", action="store_true", help="If set, re-run steps that previously ended with error.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


