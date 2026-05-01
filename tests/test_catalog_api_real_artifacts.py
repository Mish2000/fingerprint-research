from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.responses import FileResponse

import apps.api.catalog_store as catalog_store
import apps.api.demo_store as demo_store
from apps.api.main import catalog_asset, catalog_dataset_browser, catalog_datasets


def test_catalog_browser_endpoints_work_against_real_repo_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = repo_root / "data" / "processed" / "ui_assets_registry.json"

    if not registry_path.is_file():
        pytest.skip(f"real ui_assets artifacts are unavailable in this workspace snapshot: {registry_path}")

    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()

    datasets_response = catalog_datasets()
    browser_dataset = next((item.dataset for item in datasets_response.items if item.has_browser_assets), None)

    assert browser_dataset is not None, "Expected at least one dataset with browser assets in the committed artifacts."

    browser_response = catalog_dataset_browser(dataset=browser_dataset, limit=1, offset=0)
    assert browser_response.items, f"Expected at least one browser item for dataset {browser_dataset!r}."

    first_item = browser_response.items[0]
    thumbnail_response = catalog_asset(browser_dataset, first_item.asset_id, "thumbnail")
    preview_response = catalog_asset(browser_dataset, first_item.asset_id, "preview")

    assert isinstance(thumbnail_response, FileResponse)
    assert Path(thumbnail_response.path).is_file()
    assert isinstance(preview_response, FileResponse)
    assert Path(preview_response.path).is_file()
