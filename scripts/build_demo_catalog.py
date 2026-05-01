from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpbench.catalog.demo_catalog import build_catalog_bundle


if __name__ == "__main__":
    bundle = build_catalog_bundle(write_files=True)
    outdir = Path("data/samples")
    print(json.dumps({
        "catalog": str(outdir / "catalog.json"),
        "schema": str(outdir / "catalog.schema.json"),
        "report": str(outdir / "catalog.validation_report.json"),
        "validation_status": bundle["report"]["validation_status"],
        "verify_cases": bundle["catalog"]["metadata"]["total_verify_cases"],
        "identities": bundle["catalog"]["metadata"]["total_identity_records"],
    }, indent=2))
