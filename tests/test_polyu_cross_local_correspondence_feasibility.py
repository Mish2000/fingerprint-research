from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local


def test_feature_inventory_selects_final_pre_pool_spatial_stage() -> None:
    rows = local.convencoder_feature_map_inventory(width=32, input_size=384)
    selected = [row for row in rows if row["selected_for_local_descriptors"]]
    assert len(selected) == 1
    row = selected[0]
    assert row["stage_name"] == "encoder.net.3"
    assert row["spatial_height"] == 24
    assert row["spatial_width"] == 24
    assert row["channel_count"] == 256
    assert row["receptive_field_pixels"] == 76
    assert row["before_or_after_final_pooling"] == "before_final_adaptive_avg_pool"

    pooled = rows[-1]
    assert pooled["stage_name"] == "encoder.net.4"
    assert pooled["before_or_after_final_pooling"] == "after_final_adaptive_avg_pool"
    assert pooled["spatial_height"] == 1


def test_local_matcher_definitions_on_toy_descriptors() -> None:
    grid = np.eye(4, dtype=np.float32).reshape(2, 2, 4)

    chamfer = local.symmetric_local_chamfer_np(grid, grid)
    assert chamfer.score == 1.0
    assert chamfer.chamfer_a_to_b == 1.0
    assert chamfer.chamfer_b_to_a == 1.0

    mnn = local.mutual_nearest_neighbor_np(grid, grid)
    assert mnn.score == 1.0
    assert mnn.mutual_match_count == 4
    assert mnn.mean_matched_cosine == 1.0


def test_l3_fixed_translation_search_finds_shift() -> None:
    a = np.eye(9, dtype=np.float32).reshape(3, 3, 9)
    b = np.zeros_like(a)
    b[1:, 1:, :] = a[:2, :2, :]

    score = local.coarse_spatial_match_np(a, b, radius=1)
    assert score.score == 1.0
    assert score.best_dx == 1
    assert score.best_dy == 1
    assert score.best_overlap_count == 4


def test_batched_pair_scoring_preserves_order_and_mnn_score() -> None:
    grid_a = np.eye(4, dtype=np.float32).reshape(2, 2, 4)
    grid_b = grid_a.copy()
    cache = local.DescriptorCache(
        sample_uids=["a", "b"],
        uid_to_index={"a": 0, "b": 1},
        local_grids={"a": grid_a, "b": grid_b},
        global_embeddings={
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([1.0, 0.0], dtype=np.float32),
        },
        height=2,
        width=2,
        channels=4,
        selected_stage_name="encoder.net.3",
    )
    tensors = local.descriptor_cache_to_tensors(cache, torch.device("cpu"))
    df = pd.DataFrame(
        {
            "pair_id": ["p0"],
            "label": [1],
            "sample_uid_a": ["a"],
            "sample_uid_b": ["b"],
            "finger_unit_a": ["1"],
            "finger_unit_b": ["1"],
            "protocol_id": ["toy"],
        }
    )

    scores, details, _elapsed = local.score_pair_frame(
        method=local.L2_MNN,
        df=df,
        tensors=tensors,
        device=torch.device("cpu"),
        batch_size=1,
        l3_shift_radius=1,
    )
    assert scores.tolist() == [1.0]
    assert details.iloc[0]["pair_id"] == "p0"
    assert details.iloc[0]["mutual_match_count"] == 4


def _metrics(l0: float, l1: float, l2: float, l3: float, within: float = 0.75) -> pd.DataFrame:
    rows = []
    values = {
        local.L0_GLOBAL: l0,
        local.L1_CHAMFER: l1,
        local.L2_MNN: l2,
        local.L3_SPATIAL: l3,
    }
    for method, auc in values.items():
        rows.append(
            {
                "method": method,
                "stage": "inner_dev",
                "protocol": "contactless_to_contact_based",
                "roc_auc": auc,
            }
        )
        for protocol in local.WITHIN_PROTOCOLS:
            rows.append(
                {
                    "method": method,
                    "stage": "inner_dev",
                    "protocol": protocol,
                    "roc_auc": within,
                }
            )
    return pd.DataFrame(rows)


def _retrieval(l0: float, l1: float, l2: float, l3: float) -> pd.DataFrame:
    rows = []
    values = {
        local.L0_GLOBAL: l0,
        local.L1_CHAMFER: l1,
        local.L2_MNN: l2,
        local.L3_SPATIAL: l3,
    }
    for method, mrr in values.items():
        for direction in ("CL_probe_to_CB_gallery", "CB_probe_to_CL_gallery"):
            rows.append(
                {
                    "method": method,
                    "stage": "inner_dev",
                    "direction": direction,
                    "mrr": mrr,
                    "recall_at_1": 0.0,
                    "recall_at_5": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_classification_a_local_signal_present_selects_simplest() -> None:
    decision = local.classify_local_signal(
        _metrics(0.55, 0.59, 0.58, 0.57),
        _retrieval(0.10, 0.14, 0.12, 0.11),
        cfg=local.LocalAuditConfig(),
    )
    assert decision["classification"] == "A. LOCAL_SIGNAL_PRESENT"
    assert decision["official_val_gate"]["opened"] is True
    assert decision["official_val_gate"]["selected_matcher"] == local.L1_CHAMFER


def test_classification_b_when_only_coarse_geometry_reaches_gate() -> None:
    decision = local.classify_local_signal(
        _metrics(0.55, 0.56, 0.565, 0.60),
        _retrieval(0.10, 0.105, 0.11, 0.14),
        cfg=local.LocalAuditConfig(),
    )
    assert decision["classification"] == "B. GEOMETRY_HELPS_PARTIALLY"
    assert decision["official_val_gate"]["opened"] is True
    assert decision["official_val_gate"]["selected_matcher"] == local.L3_SPATIAL


def test_classification_c_keeps_official_val_closed_when_local_close_to_l0() -> None:
    decision = local.classify_local_signal(
        _metrics(0.55, 0.56, 0.565, 0.555),
        _retrieval(0.10, 0.11, 0.105, 0.10),
        cfg=local.LocalAuditConfig(),
    )
    assert decision["classification"] == "C. LOCAL_SIGNAL_WEAK"
    assert decision["official_val_gate"]["opened"] is False
    assert decision["official_val_gate"]["selected_matcher"] == ""
