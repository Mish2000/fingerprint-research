from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_DATASET = "polyu_cross"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 1 PolyU CrossFingerprint image manifest only. "
            "This scans a local dataset root and writes manifest.csv plus warnings/sanity JSON; "
            "it does not generate pairs, train models, benchmark, or copy images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help=(
            "Local PolyU CrossFingerprint dataset directory containing folders such as "
            "contactless_2d_fingerprint_images, processed_contactless_2d_fingerprint_images, "
            "and contact-based_fingerprints."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / SOURCE_DATASET / "manifest.csv",
        help="Output manifest CSV path. Default: data/manifests/polyu_cross/manifest.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image file under the dataset root has unparseable modality/metadata or unreadable dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.fpbench.datasets.polyu_cross import write_manifest

    result = write_manifest(args.dataset_root, args.output, strict=bool(args.strict))
    print(
        json.dumps(
            {
                "source_dataset": SOURCE_DATASET,
                "manifest": str(result.manifest_path),
                "warnings": str(result.warnings_path),
                "sanity_report": str(result.sanity_report_path),
                "rows": result.row_count,
                "warnings_count": result.warning_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
