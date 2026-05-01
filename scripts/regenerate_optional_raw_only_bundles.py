from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "raw_only_optional"
MANIFEST_ROOT = REPO_ROOT / "data" / "manifests"


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _dataset_commands(use_fixtures: bool) -> dict[str, list[str]]:
    if use_fixtures:
        unsw_raw = FIXTURES / "unsw_2d3d"
        polyu_raw = FIXTURES / "polyu_3d" / "3D_Fingerprint_Images_Database_V2"
        l3_raw = FIXTURES / "l3_sf_v2"
    else:
        data_raw = REPO_ROOT / "data" / "raw"
        unsw_raw = _first_existing(data_raw / "unsw_2d3d")
        polyu_raw = _first_existing(data_raw / "3D_Fingerprint_Images_Database_V2", data_raw / "polyu_3d")
        l3_raw = _first_existing(data_raw / "l3_sf_v2", data_raw / "L3SF_V2")
        missing = [
            name
            for name, root in (("unsw_2d3d", unsw_raw), ("polyu_3d", polyu_raw), ("l3_sf_v2", l3_raw))
            if root is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(
                f"Missing raw roots for: {joined}. "
                "Run on a working tree that has data/raw populated, or pass --use-fixtures for deterministic demo regeneration."
            )

    return {
        "unsw_2d3d": [
            sys.executable,
            str(REPO_ROOT / "pipelines" / "ingest" / "prepare_data_unsw_2d3d.py"),
            "--raw_root", str(unsw_raw),
            "--out_dir", str(MANIFEST_ROOT / "unsw_2d3d"),
            "--protocol", "cross_modality",
        ],
        "polyu_3d": [
            sys.executable,
            str(REPO_ROOT / "pipelines" / "ingest" / "prepare_data_polyu_3d.py"),
            "--raw_root", str(polyu_raw),
            "--out_dir", str(MANIFEST_ROOT / "polyu_3d"),
            "--protocol", "surface_only",
        ],
        "l3_sf_v2": [
            sys.executable,
            str(REPO_ROOT / "pipelines" / "ingest" / "prepare_data_l3_sf_v2.py"),
            "--raw_root", str(l3_raw),
            "--out_dir", str(MANIFEST_ROOT / "l3_sf_v2"),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate optional raw-only manifest bundles into data/manifests.")
    ap.add_argument("--use-fixtures", action="store_true", help="Use deterministic test fixtures instead of real data/raw roots.")
    ap.add_argument("--datasets", nargs="*", choices=["unsw_2d3d", "polyu_3d", "l3_sf_v2"], default=["unsw_2d3d", "polyu_3d", "l3_sf_v2"])
    args = ap.parse_args()

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    commands = _dataset_commands(use_fixtures=bool(args.use_fixtures))
    for dataset in args.datasets:
        proc = subprocess.run(commands[dataset], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.stdout + "\n" + proc.stderr)
        print(proc.stdout.strip())


if __name__ == "__main__":
    main()
