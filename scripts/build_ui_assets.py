from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpbench.catalog.demo_catalog import build_catalog_bundle
from fpbench.ui_assets.pipeline import UiAssetConfig, build_ui_assets, discover_supported_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build machine-generated UI preview assets under data/processed/<dataset>/ui_assets/.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Optional dataset names. Defaults to all datasets that have data/manifests/<dataset>/manifest.csv.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=240,
        help="Maximum number of preview items to build per dataset. Use 0 or a negative value for no limit.",
    )
    args = parser.parse_args()

    datasets = args.datasets or discover_supported_datasets(ROOT)
    max_items = None if args.max_items <= 0 else args.max_items
    registry = build_ui_assets(
        datasets,
        repo_root=ROOT,
        config=UiAssetConfig(max_items_per_dataset=max_items),
    )
    catalog_bundle = build_catalog_bundle(write_files=True)

    print(f"Built ui_assets registry for {len(registry['datasets'])} dataset(s).")
    for item in registry["datasets"]:
        print(
            f"- {item['dataset']}: {item['item_count']} item(s), "
            f"status={item['validation_status']}, index={item['index_path']}"
        )
    print(
        "Built demo catalog assets: "
        f"verify_cases={catalog_bundle['catalog']['metadata']['total_verify_cases']}, "
        f"materialized_assets={catalog_bundle['catalog']['metadata']['materialized_asset_count']}, "
        f"status={catalog_bundle['report']['validation_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
