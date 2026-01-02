#!/usr/bin/env python3
"""
Merge sharded training checkpoints into HuggingFace format (via scripts/model_merger.py),
upload each merged checkpoint to HuggingFace Hub, and cleanup disk.

Designed for VERL-style checkpoints:
  <ckpt_root>/global_step_<N>/actor/
    - model_world_size_<W>_rank_<R>.pt
    - huggingface/            (tokenizer/config saved by rank-0)

After successful merge+upload, this script can delete everything in each processed
checkpoint directory *except* the merged HuggingFace model folder, keeping the
latest checkpoint unmodified (so training can still resume from it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


WEIGHT_FILE_RE = re.compile(r".*\.(safetensors|bin)$")
SHARDED_MODEL_RE = re.compile(r"model_world_size_\d+_rank_\d+\.pt$")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _has_hf_weights(hf_dir: Path) -> bool:
    if not hf_dir.is_dir():
        return False
    try:
        for p in hf_dir.iterdir():
            if p.is_file() and WEIGHT_FILE_RE.match(p.name):
                return True
    except Exception:
        return False
    return False


def _has_sharded_model_files(local_dir: Path) -> bool:
    if not local_dir.is_dir():
        return False
    try:
        for p in local_dir.iterdir():
            if p.is_file() and SHARDED_MODEL_RE.match(p.name):
                return True
    except Exception:
        return False
    return False


def _read_latest_step_from_tracker(ckpt_root: Path) -> Optional[int]:
    tracker = ckpt_root / "checkpoint_tracker.json"
    if not tracker.exists():
        return None
    try:
        data = json.loads(tracker.read_text(encoding="utf-8"))
        v = data.get("last_global_step", None)
        return int(v) if v is not None else None
    except Exception:
        return None


def _find_model_merger(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--model-merger not found: {p}")
        return p

    here = Path(__file__).resolve()
    candidates = [
        # MNAR/merge_upload_checkpoints.py -> ../scripts/model_merger.py
        here.parent.parent / "scripts" / "model_merger.py",
        # if script is placed at repo root
        here.parent / "scripts" / "model_merger.py",
        # if running from repo root
        Path.cwd() / "scripts" / "model_merger.py",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not auto-locate scripts/model_merger.py. Provide it via --model-merger."
    )


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path  # global_step_<N>

    @property
    def actor_dir(self) -> Path:
        return self.path / "actor"

    @property
    def hf_dir_at_ckpt_root(self) -> Path:
        return self.path / "huggingface"

    @property
    def hf_dir_under_actor(self) -> Path:
        return self.actor_dir / "huggingface"


def _iter_checkpoints(ckpt_root: Path, name_re: re.Pattern[str]) -> list[Checkpoint]:
    out: list[Checkpoint] = []
    for p in ckpt_root.iterdir():
        if not p.is_dir():
            continue
        m = name_re.match(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        out.append(Checkpoint(step=step, path=p))
    out.sort(key=lambda c: c.step)
    return out


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _merge_actor_checkpoint(
    *,
    ckpt: Checkpoint,
    model_merger: Path,
    python_exe: str,
    dry_run: bool,
) -> bool:
    """
    Run model_merger.py to merge shards under <ckpt>/actor into <ckpt>/actor/huggingface.
    Returns True if HF weights exist afterwards (or already existed).
    """
    # If already merged somewhere, skip merge.
    if _has_hf_weights(ckpt.hf_dir_at_ckpt_root) or _has_hf_weights(ckpt.hf_dir_under_actor):
        return True

    if not ckpt.actor_dir.is_dir():
        _eprint(f"[step {ckpt.step}] missing actor dir: {ckpt.actor_dir}")
        return False

    if not _has_sharded_model_files(ckpt.actor_dir):
        _eprint(f"[step {ckpt.step}] no sharded model files under: {ckpt.actor_dir}")
        return False

    cmd = [
        python_exe,
        str(model_merger),
        "--local_dir",
        str(ckpt.actor_dir.resolve()),
    ]
    _run(cmd, dry_run=dry_run)

    # verify
    return _has_hf_weights(ckpt.hf_dir_under_actor)


def _upload_hf_folder(*, hf_dir: Path, repo_id: str, private: bool, hf_token: Optional[str], dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN upload: {hf_dir} -> {repo_id} (private={private})")
        return

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for upload. Install it or run in the training env."
        ) from e

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(hf_dir), repo_type="model")


def _ensure_hf_at_ckpt_root(*, ckpt: Checkpoint, dry_run: bool) -> Optional[Path]:
    """
    Ensure merged HF folder ends up at <ckpt>/huggingface (so cleanup can keep only that).
    Returns the kept HF dir path, or None if missing.
    """
    if _has_hf_weights(ckpt.hf_dir_at_ckpt_root):
        return ckpt.hf_dir_at_ckpt_root

    if _has_hf_weights(ckpt.hf_dir_under_actor):
        dst = ckpt.hf_dir_at_ckpt_root
        src = ckpt.hf_dir_under_actor
        if dst.exists():
            # If dst exists but doesn't have weights, remove it first to allow move.
            if dry_run:
                print(f"DRY-RUN: would remove existing {dst} then move {src} -> {dst}")
                return dst
            shutil.rmtree(dst, ignore_errors=True)

        if dry_run:
            print(f"DRY-RUN move: {src} -> {dst}")
        else:
            shutil.move(str(src), str(dst))
        return dst

    return None


def _cleanup_checkpoint_dir(*, ckpt: Checkpoint, keep_hf_dir: Path, yes: bool, dry_run: bool) -> None:
    """
    Delete everything under <ckpt> except keep_hf_dir (<ckpt>/huggingface).
    """
    if not yes:
        print(f"[step {ckpt.step}] --yes not set; skip deletion. (would keep only {keep_hf_dir})")
        return

    # Delete all siblings of keep_hf_dir.
    for child in ckpt.path.iterdir():
        if child.resolve() == keep_hf_dir.resolve():
            continue
        if dry_run:
            print(f"DRY-RUN delete: {child}")
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except Exception as e:
            _eprint(f"[step {ckpt.step}] failed to delete {child}: {e}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-root",
        required=True,
        type=str,
        help="Checkpoint root directory containing global_step_* folders (e.g., checkpoints/<proj>/<exp>).",
    )
    parser.add_argument(
        "--ckpt-regex",
        type=str,
        default=r"^global_step_(\d+)$",
        help=r"Regex to match checkpoint folder name; must capture step as group(1). Default: ^global_step_(\d+)$",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=1,
        help="How many latest checkpoints to keep unmerged/unmodified. Default: 1.",
    )
    parser.add_argument(
        "--role-subdir",
        type=str,
        default="actor",
        help="Role subdir under each checkpoint to merge (usually actor). Default: actor.",
    )
    parser.add_argument(
        "--model-merger",
        type=str,
        default=None,
        help="Path to scripts/model_merger.py (auto-detected if omitted).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run model_merger.py. Default: current python.",
    )
    parser.add_argument(
        "--hf-prefix",
        type=str,
        default="",
        help='If set, upload merged model to repo_id = hf_prefix + str(step). Example: "nicoledy/qwen2.5-math-base-limr-step-"',
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create HuggingFace repos as private. (Requires permission.)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (optional). If omitted, huggingface_hub will use local cached token / env.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete checkpoint files after merge/upload.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files during cleanup. Without this flag, cleanup is a no-op.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned deletions, but do not execute.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    ckpt_root = Path(args.ckpt_root).expanduser().resolve()
    if not ckpt_root.is_dir():
        _eprint(f"--ckpt-root is not a directory: {ckpt_root}")
        return 2

    # We only support VERL layout for now: <ckpt>/actor
    name_re = re.compile(args.ckpt_regex)
    ckpts = _iter_checkpoints(ckpt_root, name_re=name_re)
    if not ckpts:
        _eprint(f"No checkpoints matched under {ckpt_root} with regex: {args.ckpt_regex}")
        return 1

    # Optionally use tracker to sanity-check what's considered "latest".
    tracker_latest = _read_latest_step_from_tracker(ckpt_root)
    if tracker_latest is not None:
        print(f"Found checkpoint_tracker.json latest_global_step={tracker_latest}")

    keep_latest = max(0, int(args.keep_latest))
    to_process = ckpts[:-keep_latest] if keep_latest > 0 else ckpts
    skipped = ckpts[-keep_latest:] if keep_latest > 0 else []

    print(f"Checkpoint root: {ckpt_root}")
    print(f"Matched checkpoints: {len(ckpts)}; will process: {len(to_process)}; will keep latest: {len(skipped)}")
    if skipped:
        print("Keep-latest steps (no merge/cleanup): " + ", ".join(str(c.step) for c in skipped))

    model_merger = _find_model_merger(args.model_merger)
    print(f"Using model_merger: {model_merger}")

    failures: list[int] = []
    for ckpt in to_process:
        # Allow overriding role_subdir: we only merge actor by default.
        if args.role_subdir != "actor":
            ckpt = Checkpoint(step=ckpt.step, path=ckpt.path)  # reuse but we will compute dirs manually below

        # Derive actor dir dynamically (based on --role-subdir)
        actor_dir = ckpt.path / args.role_subdir
        hf_under_role = actor_dir / "huggingface"
        hf_at_root = ckpt.path / "huggingface"

        print(f"\n=== step {ckpt.step} ===")
        print(f"ckpt: {ckpt.path}")

        # Merge (only if we still have shards)
        merged_ok = True
        if _has_hf_weights(hf_at_root) or _has_hf_weights(hf_under_role):
            print(f"[step {ckpt.step}] HF weights already present; skip merge.")
        else:
            # Temporarily construct a Checkpoint-like view for actor_dir required by model_merger
            tmp_ckpt = Checkpoint(step=ckpt.step, path=ckpt.path)
            # monkeypatch style: call merger using actor_dir
            if not actor_dir.is_dir():
                _eprint(f"[step {ckpt.step}] missing role dir: {actor_dir}")
                merged_ok = False
            elif not _has_sharded_model_files(actor_dir):
                _eprint(f"[step {ckpt.step}] no sharded model files under: {actor_dir}")
                merged_ok = False
            else:
                cmd = [args.python, str(model_merger), "--local_dir", str(actor_dir.resolve())]
                try:
                    _run(cmd, dry_run=args.dry_run)
                except subprocess.CalledProcessError as e:
                    _eprint(f"[step {ckpt.step}] merge failed: {e}")
                    merged_ok = False

        # Determine HF directory to upload/keep.
        hf_dir: Optional[Path] = None
        if _has_hf_weights(hf_at_root):
            hf_dir = hf_at_root
        elif _has_hf_weights(hf_under_role):
            hf_dir = hf_under_role
        else:
            hf_dir = None

        if not merged_ok or hf_dir is None:
            failures.append(ckpt.step)
            continue

        # Upload (optional)
        if args.hf_prefix:
            repo_id = f"{args.hf_prefix}{ckpt.step}"
            try:
                _upload_hf_folder(
                    hf_dir=hf_dir,
                    repo_id=repo_id,
                    private=bool(args.private),
                    hf_token=args.hf_token,
                    dry_run=args.dry_run,
                )
                print(f"[step {ckpt.step}] uploaded -> {repo_id}")
            except Exception as e:
                _eprint(f"[step {ckpt.step}] upload failed: {e}")
                failures.append(ckpt.step)
                continue

        # Cleanup (optional)
        if args.no_cleanup:
            print(f"[step {ckpt.step}] --no-cleanup set; keep checkpoint files.")
            continue

        # Ensure HF folder ends at <ckpt>/huggingface so we can delete everything else.
        keep_hf = None
        if _has_hf_weights(hf_at_root):
            keep_hf = hf_at_root
        else:
            # move <ckpt>/<role>/huggingface -> <ckpt>/huggingface
            if args.dry_run:
                print(f"DRY-RUN move: {hf_under_role} -> {hf_at_root}")
                keep_hf = hf_at_root
            else:
                try:
                    if hf_at_root.exists():
                        shutil.rmtree(hf_at_root, ignore_errors=True)
                    shutil.move(str(hf_under_role), str(hf_at_root))
                    keep_hf = hf_at_root
                except Exception as e:
                    _eprint(f"[step {ckpt.step}] failed to move hf folder to ckpt root: {e}")
                    failures.append(ckpt.step)
                    continue

        _cleanup_checkpoint_dir(ckpt=ckpt, keep_hf_dir=keep_hf, yes=bool(args.yes), dry_run=args.dry_run)

    if failures:
        _eprint(f"\nDone with failures at steps: {sorted(set(failures))}")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


