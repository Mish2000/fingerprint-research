from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnostics import run_polyu_cross_flare_external_baseline as flare


def test_staged_filename_preserves_supported_suffix_and_uses_sample_uid() -> None:
    assert flare.staged_filename("uid123", r"C:\x\raw name.JPG") == "uid123.jpg"
    assert flare.staged_filename("uid123", "no_suffix") == "uid123.png"
    assert flare.staged_filename("uid123", "scan.wsq") == "uid123.png"


def test_build_adapter_stage_uses_reversible_links(tmp_path: Path) -> None:
    src = tmp_path / "src.bmp"
    src.write_bytes(b"not-an-image-but-linkable")
    images = pd.DataFrame(
        [
            {
                "sample_uid": "polyu_uid_a",
                "finger_unit_id": "11",
                "modality": flare.CONTACTLESS,
                "session_id": "session_1",
                "resolved_path": str(src),
            }
        ]
    )
    stage_data = flare.StageData(
        stage="inner_dev",
        split="train",
        images=images,
        identity_ids=["11"],
        pair_bundle={},
        retrieval_table=pd.DataFrame(),
        resolved_root=None,
        pair_counts={},
    )

    adapter = flare.build_adapter_stage(stage_data=stage_data, stage_dir=tmp_path / "stage", allow_copy=False)

    assert len(adapter) == 2
    assert set(adapter["role"]) == {"query", "gallery"}
    assert not adapter["copied_biometric_image"].any()
    for rel in adapter["staged_relative_path"]:
        assert (tmp_path / "stage" / rel).exists()


def test_score_pair_frame_from_matrix_joins_by_sample_uid_pkl() -> None:
    matrix = pd.DataFrame(
        [[0.9, 0.1], [0.2, 0.8]],
        index=["a.pkl", "b.pkl"],
        columns=["x.pkl", "y.pkl"],
    )
    pairs = pd.DataFrame(
        [
            {"sample_uid_a": "a", "sample_uid_b": "x", "label": 1},
            {"sample_uid_a": "b", "sample_uid_b": "y", "label": 0},
            {"sample_uid_a": "missing", "sample_uid_b": "x", "label": 0},
        ]
    )

    scores = flare.score_pair_frame_from_matrix(pairs, matrix)

    assert np.allclose(scores[:2], [0.9, 0.8])
    assert math.isnan(float(scores[2]))


def _metrics(condition: str, clcb_auc: float) -> pd.DataFrame:
    rows = []
    for protocol, auc in {
        flare.CLCB: clcb_auc,
        flare.CLCL_SAME: 0.82,
        flare.CLCL_CROSS: 0.66,
        flare.CBCB_SAME: 0.80,
        flare.CBCB_CROSS: 0.65,
    }.items():
        rows.append(
            {
                "condition": condition,
                "stage": "inner_dev",
                "protocol": protocol,
                "roc_auc": auc,
                "pair_count": 10,
                "failed_count": 0,
            }
        )
    return pd.DataFrame(rows)


def _retrieval(condition: str, cl_mrr: float, cb_mrr: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition": condition,
                "stage": "inner_dev",
                "direction": "CL_probe_to_CB_gallery",
                "mrr": cl_mrr,
            },
            {
                "condition": condition,
                "stage": "inner_dev",
                "direction": "CB_probe_to_CL_gallery",
                "mrr": cb_mrr,
            },
        ]
    )


def test_inner_dev_gate_requires_auc_retrieval_within_and_no_major_failures() -> None:
    cfg = flare.FlareConfig()
    metrics = pd.concat([_metrics(flare.E1, 0.72), _metrics(flare.E2, 0.55)], ignore_index=True)
    retrieval = pd.concat([_retrieval(flare.E1, 0.20, 0.16), _retrieval(flare.E2, 0.20, 0.16)], ignore_index=True)
    failures = pd.DataFrame(columns=["condition", "stage", "major_issue"])

    gates = flare.condition_gate_rows(metrics, retrieval, failures, cfg)
    by_condition = {row["condition"]: row for row in gates}

    assert by_condition[flare.E1]["passed"] is True
    assert by_condition[flare.E2]["passed"] is False
    assert by_condition[flare.E2]["criteria"]["clcb_auc_at_least_0_70"] is False


def test_selection_prefers_regression_pose_on_small_auc_tie() -> None:
    cfg = flare.FlareConfig(selection_auc_tie_margin=0.01)
    metrics = pd.concat([_metrics(flare.E1, 0.721), _metrics(flare.E2, 0.725)], ignore_index=True)
    gate_rows = [
        {"condition": flare.E1, "passed": True},
        {"condition": flare.E2, "passed": True},
    ]

    assert flare.select_val_condition(gate_rows, metrics, cfg) == flare.E1


def test_classification_c_when_external_baseline_is_weak_and_clean() -> None:
    cfg = flare.FlareConfig()
    metrics = pd.concat([_metrics(flare.E1, 0.56), _metrics(flare.E2, 0.55)], ignore_index=True)
    retrieval = pd.concat([_retrieval(flare.E1, 0.10, 0.10), _retrieval(flare.E2, 0.10, 0.10)], ignore_index=True)
    failures = pd.DataFrame(columns=["condition", "stage", "major_issue"])

    decision = flare.classify_decision(
        metrics=metrics,
        retrieval=retrieval,
        failures=failures,
        val_metrics=pd.DataFrame(),
        cfg=cfg,
        dependency_failures=[],
        val_condition=None,
    )

    assert decision["classification"] == "C. EXTERNAL_BASELINE_ALSO_FAILS"
    assert decision["official_val_gate"]["opened"] is False
