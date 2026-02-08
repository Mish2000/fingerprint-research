"""
Week 11 (Step 3): Discover / record evaluation script CLIs.

Purpose:
- Capture `--help` output for the evaluation entry points we intend to automate in run_all.py
- Write results to reports/week11/discover_eval.log

This step does NOT run evaluations yet.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "reports" / "week11"
LOG_PATH = OUTDIR / "discover_eval.log"

CANDIDATES = [
    Path("scripts/week05/evaluate.py"),
    Path("scripts/week08/eval_dedicated.py"),
]


def run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    combined = ""
    if p.stdout:
        combined += p.stdout
    if p.stderr:
        combined += ("\n" if combined and not combined.endswith("\n") else "") + p.stderr
    return p.returncode, combined.strip()


def log(line: str) -> None:
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("=== Week 11 Step 3: discover evaluation interfaces ===")
    log(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    log(f"repo_root    : {REPO_ROOT}")
    log(f"python       : {sys.executable}")
    log("")

    missing = [p for p in CANDIDATES if not (REPO_ROOT / p).exists()]
    if missing:
        log("ERROR: Missing candidate scripts:")
        for p in missing:
            log(f"  - {p.as_posix()}")
        return 2

    any_fail = False

    for rel in CANDIDATES:
        abs_path = REPO_ROOT / rel
        log(f"--- {rel.as_posix()} --help ---")
        rc, out = run([sys.executable, str(abs_path), "--help"])
        log(f"exit_code: {rc}")
        log(out if out else "(no output)")
        log("")
        if rc != 0:
            any_fail = True

    if any_fail:
        log("DONE with WARN: One or more --help commands returned non-zero.")
        log("This is still useful: paste this log and we will adapt accordingly.")
        return 1

    log("DONE: All --help outputs captured successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())