from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "regenerate_optional_raw_only_bundles.py"),
        "--use-fixtures",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.stdout + "\n" + proc.stderr)
    print(proc.stdout)


if __name__ == "__main__":
    main()
