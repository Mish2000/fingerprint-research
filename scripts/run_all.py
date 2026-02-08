"""
Week 11 automation entry point.

Step 2 (already done):
- Preflight checks for required processed data files
- Creates reports/week11/
- Writes run_manifest.json and run.log

Step 4 (this step):
- Optional: run exactly ONE evaluation via scripts/week05/evaluate.py
  and write outputs into reports/week11/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # Python 3.8+
    from importlib.metadata import version as pkg_version  # type: ignore
except Exception:
    pkg_version = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "reports" / "week11"

REQUIRED_FILES = [
    Path("data/processed/nist_sd300b/split.json"),
    Path("data/processed/nist_sd300b/stats.json"),
]

OPTIONAL_FILES = [
    Path("data/processed/nist_sd300b/pairs_split_build.meta.json"),
]

EVALUATE_PY = Path("scripts/week05/evaluate.py")


def resolve_dedicated_ckpt(user_value: str) -> str:
    """
    Resolves the dedicated descriptor checkpoint.
    - If user_value != 'auto': return it as-is.
    - If 'auto': search under reports/** for patch_descriptor_ckpt.pth
    """
    if user_value and user_value.lower() != "auto":
        return user_value

    reports_dir = REPO_ROOT / "reports"
    # Most common expected filename (per your error message)
    hits = list(reports_dir.rglob("patch_descriptor_ckpt.pth"))

    # Fallback: any .pth in a patch_descriptor/final folder
    if not hits:
        hits = list(reports_dir.rglob("patch_descriptor/final/*.pth"))

    if not hits:
        raise FileNotFoundError(
            "Could not auto-resolve dedicated checkpoint. "
            "Expected something like reports/week06+07/patch_descriptor/final/patch_descriptor_ckpt.pth"
        )

    # Pick the most recently modified checkpoint
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(hits[0])


@dataclass
class GitInfo:
    commit: Optional[str]
    is_dirty: Optional[bool]
    branch: Optional[str]
    error: Optional[str]


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def run_cmd_stream(cmd: List[str], cwd: Path) -> int:
    """
    Stream subprocess output to console (and let caller also write a log line).
    """
    p = subprocess.run(cmd, cwd=str(cwd))
    return p.returncode


def get_git_info(repo_root: Path) -> GitInfo:
    try:
        rc, out, err = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if rc != 0:
            return GitInfo(None, None, None, f"git rev-parse failed: {err or out}")

        commit = out

        rc, out, err = run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
        if rc != 0:
            return GitInfo(commit, None, None, f"git status failed: {err or out}")
        is_dirty = len(out.strip()) > 0

        rc, out, err = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        if rc != 0:
            return GitInfo(commit, is_dirty, None, f"git branch failed: {err or out}")
        branch = out.strip()

        return GitInfo(commit, is_dirty, branch, None)
    except FileNotFoundError:
        return GitInfo(None, None, None, "git not found on PATH")
    except Exception as e:
        return GitInfo(None, None, None, f"git info error: {e}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_pkg_version(name: str) -> Optional[str]:
    if pkg_version is None:
        return None
    try:
        return pkg_version(name)
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_log_line(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def build_eval_cmd(
        *,
        outdir: Path,
        method: str,
        split: str,
        limit: int,
        ensure_pairs: bool,
        dedicated_ckpt: str
) -> List[str]:
    """
    Build a deterministic evaluate.py command that writes ALL artifacts into outdir.
    This runs ONLY one method/split per invocation.
    """
    summary_csv = outdir / "results_summary.csv"
    out_scores = outdir / f"scores_{method}_{split}.csv"
    out_roc = outdir / f"roc_{method}_{split}.png"
    out_run_meta = outdir / f"run_{method}_{split}.meta.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / EVALUATE_PY),
        "--method",
        method,
        "--split",
        split,
        "--limit",
        str(limit),
        "--summary_csv",
        str(summary_csv),
        "--out_scores",
        str(out_scores),
        "--out_roc",
        str(out_roc),
        "--out_run_meta",
        str(out_run_meta),
    ]

    if ensure_pairs:
        cmd.append("--ensure_pairs")

    # Canonical configs (match what you’ve been using in the project)
    if method == "classic_v2":
        cmd += [
            "--detector",
            "gftt_orb",
            "--score_mode",
            "inliers_over_k",
            "--nfeatures",
            "1500",
            "--long_edge",
            "512",
            "--ratio",
            "0.75",
            "--ransac_thresh",
            "4.0",
        ]
    elif method == "dl_quick":
        cmd += [
            "--backbone",
            "resnet50",
            "--no_mask",
        ]
    elif method == "dedicated":
        cmd += [
            "--dedicated_patch",
            "48",
            "--dedicated_max_kpts",
            "800",
            "--dedicated_mask_cov_thr",
            "0.7",
            "--dedicated_max_matches",
            "200",
            "--dedicated_ransac_thresh",
            "4.0",
            "--dedicated_ckpt",
            resolve_dedicated_ckpt(dedicated_ckpt),
        ]

    return cmd

def render_results_md(summary_csv: Path, summary_md: Path) -> None:
    import csv

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("results_summary.csv exists but is empty")

    # Preferred display order
    split_order = {"val": 0, "test": 1, "train": 2}
    method_order = {"classic_v2": 0, "dl_quick": 1, "dedicated": 2}

    def k(r):
        return (split_order.get(r.get("split", ""), 99), method_order.get(r.get("method", ""), 99), r.get("method", ""))

    rows.sort(key=k)

    cols = [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
    ]

    # Some older summaries may not have all cols; keep only existing
    existing_cols = [c for c in cols if c in rows[0]]

    # Write markdown table
    lines = []
    lines.append("# Week 11 Results Summary\n")
    lines.append("| " + " | ".join(existing_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(existing_cols)) + " |")

    for r in rows:
        vals = []
        for c in existing_cols:
            v = r.get(c, "")
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Output directory for week11 artifacts")

    # Step 4: run exactly one eval
    ap.add_argument("--eval_one", action="store_true", help="Run a single evaluate.py job (one method/split).")
    ap.add_argument("--method", type=str, default="classic_v2", choices=["classic_v2", "dl_quick", "dedicated"])
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ensure_pairs", action="store_true", help="Pass --ensure_pairs to evaluate.py")
    ap.add_argument("--dedicated_ckpt", type=str, default="auto", help="Path to descriptor ckpt (.pth) or 'auto'")
    ap.add_argument("--eval_all", action="store_true", help="Run all methods on val+test (Week 11 main mode).")
    ap.add_argument("--splits", type=str, default="val,test", help="Comma-separated splits to run (default: val,test)")
    ap.add_argument("--methods", type=str, default="classic_v2,dl_quick,dedicated",
                    help="Comma-separated methods to run")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    log_path = outdir / "run.log"
    # start fresh each run
    if log_path.exists():
        log_path.unlink()

    def log(msg: str) -> None:
        print(msg)
        write_log_line(log_path, msg)

    log("=== Week 11 run_all.py ===")
    log(f"Repo root: {REPO_ROOT}")
    log(f"Outdir   : {outdir}")

    # Preflight required files
    missing_required: List[Path] = []
    for rel in REQUIRED_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            missing_required.append(rel)

    if missing_required:
        log("ERROR: Missing required processed-data files:")
        for rel in missing_required:
            log(f"  - {rel.as_posix()}")
        log("Fix: Ensure you ran the data processing step that creates data/processed/nist_sd300b/*.json")
        return 2

    # Optional files (warn only)
    missing_optional: List[Path] = []
    for rel in OPTIONAL_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            missing_optional.append(rel)

    if missing_optional:
        log("WARN: Missing optional (but recommended) metadata files:")
        for rel in missing_optional:
            log(f"  - {rel.as_posix()}")

    # Collect hashes
    hashes: Dict[str, str] = {}
    for rel in REQUIRED_FILES + [r for r in OPTIONAL_FILES if (REPO_ROOT / r).exists()]:
        p = REPO_ROOT / rel
        hashes[rel.as_posix()] = sha256_file(p)

    # Git + environment info
    git = get_git_info(REPO_ROOT)
    ts = datetime.now(timezone.utc).isoformat()

    pkgs = {
        "numpy": safe_pkg_version("numpy"),
        "opencv-python": safe_pkg_version("opencv-python"),
        "torch": safe_pkg_version("torch"),
        "torchvision": safe_pkg_version("torchvision"),
        "fastapi": safe_pkg_version("fastapi"),
        "uvicorn": safe_pkg_version("uvicorn"),
        "httpx": safe_pkg_version("httpx"),
    }

    manifest = {
        "timestamp_utc": ts,
        "argv": sys.argv,
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env": {"conda_prefix": os.environ.get("CONDA_PREFIX"), "venv": os.environ.get("VIRTUAL_ENV")},
        "git": {"commit": git.commit, "branch": git.branch, "is_dirty": git.is_dirty, "error": git.error},
        "file_hashes_sha256": hashes,
        "package_versions": pkgs,
        "stage": "week11_step4_eval_one_optional",
    }

    manifest_path = outdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log("OK: Preflight passed.")
    log(f"OK: Wrote {manifest_path.relative_to(REPO_ROOT).as_posix()}")
    log(f"OK: Wrote {log_path.relative_to(REPO_ROOT).as_posix()}")

    # Step 4: run one evaluation
    if args.eval_one:
        eval_py_abs = REPO_ROOT / EVALUATE_PY
        if not eval_py_abs.exists():
            log(f"ERROR: Missing {EVALUATE_PY.as_posix()}")
            return 3

        if args.method == "dedicated":
            log(f"[DEDICATED] using ckpt: {resolve_dedicated_ckpt(args.dedicated_ckpt)}")

        cmd = build_eval_cmd(
            outdir=outdir,
            method=args.method,
            split=args.split,
            limit=args.limit,
            ensure_pairs=args.ensure_pairs,
            dedicated_ckpt=args.dedicated_ckpt,
        )

        log("")
        log("=== EVAL_ONE ===")
        log("[RUN] " + " ".join(cmd))
        rc = run_cmd_stream(cmd, cwd=REPO_ROOT)
        log(f"[EVAL_ONE] exit_code={rc}")

        # Minimal post-checks: expected artifacts exist
        expected = [
            outdir / "results_summary.csv",
            outdir / f"scores_{args.method}_{args.split}.csv",
            outdir / f"roc_{args.method}_{args.split}.png",
            outdir / f"run_{args.method}_{args.split}.meta.json",
        ]
        missing = [p for p in expected if not p.exists()]
        if missing:
            log("ERROR: Missing expected outputs:")
            for p in missing:
                log(f"  - {p.relative_to(REPO_ROOT).as_posix()}")
            return 4

        log("OK: EVAL_ONE outputs created successfully.")
        log("Done (Step 4).")

    # Step 10: run all evaluations
    if args.eval_all:
        eval_py_abs = REPO_ROOT / EVALUATE_PY
        if not eval_py_abs.exists():
            log(f"ERROR: Missing {EVALUATE_PY.as_posix()}")
            return 3

        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

        # Start fresh to avoid duplicates
        summary_csv = outdir / "results_summary.csv"
        summary_md = outdir / "results_summary.md"
        if summary_csv.exists():
            summary_csv.unlink()
        if summary_md.exists():
            summary_md.unlink()

        log("")
        log("=== EVAL_ALL ===")
        log(f"methods: {methods}")
        log(f"splits : {splits}")

        for split in splits:
            for method in methods:
                if method == "dedicated":
                    ckpt = resolve_dedicated_ckpt(args.dedicated_ckpt)
                    log(f"[DEDICATED] using ckpt: {ckpt}")
                cmd = build_eval_cmd(
                    outdir=outdir,
                    method=method,
                    split=split,
                    limit=args.limit,
                    ensure_pairs=args.ensure_pairs,
                    dedicated_ckpt=args.dedicated_ckpt,
                )
                log("[RUN] " + " ".join(cmd))
                rc = run_cmd_stream(cmd, cwd=REPO_ROOT)
                log(f"[EVAL_ALL] method={method} split={split} exit_code={rc}")
                if rc != 0:
                    log("ERROR: EVAL_ALL stopped due to non-zero exit code.")
                    return 4

        # Render markdown summary
        render_results_md(summary_csv, summary_md)
        log(f"OK: Wrote {summary_md.relative_to(REPO_ROOT).as_posix()}")
        log("Done (Step 10).")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
