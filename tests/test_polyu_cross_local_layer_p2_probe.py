from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.diagnostics import run_polyu_cross_local_layer_p2_probe as probe


def test_stage_inventory_predeclares_e_m_f_and_pooling_rules() -> None:
    inv = probe.feature_stage_inventory(width=32, input_size=384)
    assert inv["stage_label"].tolist() == ["E", "M", "F"]
    assert inv["stage_name"].tolist() == ["encoder.net.0", "encoder.net.2", "encoder.net.3"]
    assert inv["native_spatial_height"].tolist() == [192, 48, 24]
    assert inv["channels"].tolist() == [32, 128, 256]
    assert inv["descriptor_grid_operation"].tolist() == [
        "adaptive_avg_pool2d((24,24))",
        "adaptive_avg_pool2d((24,24))",
        "preserve_native_24x24",
    ]
    assert inv["descriptor_count"].tolist() == [576, 576, 576]


def test_p2_robust_intensity_norm_maps_percentiles_and_keeps_degenerate_identity() -> None:
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = probe.p2_robust_intensity_norm(img)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255

    const = np.full((8, 8), 17, dtype=np.uint8)
    assert np.array_equal(probe.p2_robust_intensity_norm(const), const)


def test_normalize_feature_grid_pools_larger_maps_and_refuses_upsampling() -> None:
    x = torch.ones((2, 3, 48, 48), dtype=torch.float32)
    pooled, op = probe.normalize_feature_grid(x)
    assert pooled.shape == (2, 24, 24, 3)
    assert op == "adaptive_avg_pool2d((24,24))"
    norms = torch.linalg.vector_norm(pooled, dim=3)
    assert torch.allclose(norms, torch.ones_like(norms))

    native = torch.ones((1, 5, 24, 24), dtype=torch.float32)
    kept, op = probe.normalize_feature_grid(native)
    assert kept.shape == (1, 24, 24, 5)
    assert op == "preserve_native_24x24"

    with pytest.raises(probe.LayerP2ProbeError):
        probe.normalize_feature_grid(torch.ones((1, 5, 12, 12), dtype=torch.float32))


def _metrics(values: dict[str, float]) -> pd.DataFrame:
    rows = []
    for condition_id, auc in values.items():
        stage, prep = condition_id.split("x")
        rows.append(
            {
                "condition_id": condition_id,
                "stage_label": stage,
                "stage_name": {"E": "encoder.net.0", "M": "encoder.net.2", "F": "encoder.net.3"}[stage],
                "preprocess_condition": prep,
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows)


def _retrieval(mrr: float = 0.10, good_condition: str | None = None) -> pd.DataFrame:
    rows = []
    for stage in ("E", "M", "F"):
        for prep in (probe.RAW, probe.P2):
            cid = f"{stage}x{prep}"
            for direction, baseline in probe.BASELINE_RETRIEVAL.items():
                is_good = cid == good_condition
                rows.append(
                    {
                        "condition_id": cid,
                        "direction": direction,
                        "mrr": baseline["mrr"] + 0.04 if is_good else mrr,
                        "recall_at_1": baseline["recall_at_1"] if is_good else 0.0,
                        "recall_at_5": baseline["recall_at_5"] if is_good else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _diagnostics(good_condition: str | None = None) -> pd.DataFrame:
    rows = []
    for stage in ("E", "M", "F"):
        for prep in (probe.RAW, probe.P2):
            cid = f"{stage}x{prep}"
            separated = cid == good_condition
            rows.append(
                {
                    "condition_id": cid,
                    "pair_group": "genuine",
                    "mean_matched_cosine_mean": 0.90 if separated else 0.88,
                    "mutual_match_count_mean": 3.0 if separated else 2.3,
                }
            )
            rows.append(
                {
                    "condition_id": cid,
                    "pair_group": "impostor",
                    "mean_matched_cosine_mean": 0.86 if separated else 0.879,
                    "mutual_match_count_mean": 2.0 if separated else 2.25,
                }
            )
    return pd.DataFrame(rows)


def test_classification_a_requires_auc_retrieval_and_diagnostic_separation() -> None:
    values = {
        "ExRAW": 0.50,
        "ExP2": 0.51,
        "MxRAW": 0.52,
        "MxP2": 0.57,
        "FxRAW": 0.5119,
        "FxP2": 0.50,
    }
    decision = probe.classify_probe(
        _metrics(values),
        _retrieval(good_condition="MxP2"),
        _diagnostics(good_condition="MxP2"),
        cfg=probe.ProbeConfig(),
    )
    assert decision["classification"] == "A. FROZEN_LOCAL_SIGNAL_PRESENT"
    assert decision["official_val"]["opened"] is False


def test_classification_b_photometric_help_only() -> None:
    values = {
        "ExRAW": 0.50,
        "ExP2": 0.53,
        "MxRAW": 0.51,
        "MxP2": 0.54,
        "FxRAW": 0.5119,
        "FxP2": 0.52,
    }
    decision = probe.classify_probe(
        _metrics(values),
        _retrieval(),
        _diagnostics(),
        cfg=probe.ProbeConfig(),
    )
    assert decision["classification"] == "B. PHOTOMETRIC_HELP_ONLY"


def test_classification_c_when_all_close_and_diagnostics_identical() -> None:
    values = {
        "ExRAW": 0.50,
        "ExP2": 0.505,
        "MxRAW": 0.515,
        "MxP2": 0.512,
        "FxRAW": 0.5119,
        "FxP2": 0.509,
    }
    decision = probe.classify_probe(
        _metrics(values),
        _retrieval(),
        _diagnostics(),
        cfg=probe.ProbeConfig(),
    )
    assert decision["classification"] == "C. FROZEN_LOCAL_FEATURES_NOT_CROSS_MODAL"


def test_classification_d_when_p2_improvement_is_not_consistent_across_stages() -> None:
    values = {
        "ExRAW": 0.45,
        "ExP2": 0.49,
        "MxRAW": 0.50,
        "MxP2": 0.51,
        "FxRAW": 0.5119,
        "FxP2": 0.50,
    }
    decision = probe.classify_probe(
        _metrics(values),
        _retrieval(),
        _diagnostics(),
        cfg=probe.ProbeConfig(),
    )
    assert decision["classification"] == "D. MIXED_OR_INCONCLUSIVE"
