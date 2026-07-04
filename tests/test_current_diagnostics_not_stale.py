from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_ROOT = PROJECT_ROOT / "artifacts" / "reports" / "diagnostics"
STALE_MESSAGE = "These diagnostics are stale and do not correspond to the current canonical Fusion v2 statistical run."
STALE_DIR_NAMES = (
    "sourceafis_sift_quality_deep_fusion_v2_failure_taxonomy",
    "true_accept_failures_across_methods",
)
CURRENT_DIRS = (
    DIAGNOSTICS_ROOT / "sourceafis_sift_quality_deep_fusion_v2_current_failure_taxonomy",
    DIAGNOSTICS_ROOT / "true_accept_failures_across_methods_current",
)


def test_legacy_diagnostics_are_quarantined_or_marked_stale() -> None:
    quarantine = DIAGNOSTICS_ROOT / "legacy_stale_20260629"
    assert quarantine.exists()
    assert STALE_MESSAGE in (quarantine / "README_STALE.md").read_text(encoding="utf-8")

    for name in STALE_DIR_NAMES:
        original = DIAGNOSTICS_ROOT / name
        quarantined = quarantine / name
        if original.exists():
            marker = original / "README_STALE.md"
            assert marker.exists(), f"{original} exists and must carry a stale marker"
            assert STALE_MESSAGE in marker.read_text(encoding="utf-8")
        else:
            assert quarantined.exists(), f"{name} must be present in the stale quarantine"


def test_current_manifests_point_to_current_outputs_not_legacy_paths() -> None:
    for current_dir in CURRENT_DIRS:
        manifest_path = current_dir / "current_diagnostics_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["schema_version"].startswith("current_")
        assert "_current" in manifest["output_dir"]
        assert "legacy_stale_20260629" in manifest["stale_legacy_quarantine"]
        assert manifest["target_far"] == 0.01
        assert manifest["canonical_method"] == "sourceafis_sift_quality_deep_fusion_v2"
        assert manifest["threshold_protocol"] == "Each method threshold is computed from its own VAL negatives only."

        for counts in manifest["test_count_validation"].values():
            assert counts == {"pairs": 3556, "positives": 889, "negatives": 2667}
            assert counts["pairs"] != 2844
            assert counts["positives"] != 711
