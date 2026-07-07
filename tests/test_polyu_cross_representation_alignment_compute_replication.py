from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.diagnostics import run_polyu_cross_representation_alignment_compute_replication as rep


def test_choose_global_batch_prefers_64_on_two_gpus_and_32_fallback_on_one() -> None:
    two = rep.choose_global_batch(world_size=2, target=64, fallback=32)
    assert two["actual_global_batch_identities"] == 64
    assert two["actual_local_batch_identities"] == 32
    assert two["fallback_reason"] == ""

    one = rep.choose_global_batch(world_size=1, target=64, fallback=32)
    assert one["actual_global_batch_identities"] == 32
    assert one["actual_local_batch_identities"] == 32
    assert one["fallback_reason"] == "fallback_global_batch_32_world_size_lt_2"
    assert one["gradient_accumulation_used"] is False


def test_global_batch_slicing_assigns_disjoint_rank_chunks() -> None:
    batch = {
        "identity_ids": [str(i) for i in range(8)],
        "cl_uids": [f"cl{i}" for i in range(8)],
        "cb_uids": [f"cb{i}" for i in range(8)],
    }
    rank0 = rep.local_slice_for_rank(batch, rank=0, local_batch_identities=4)
    rank1 = rep.local_slice_for_rank(batch, rank=1, local_batch_identities=4)
    assert rank0["identity_ids"] == ["0", "1", "2", "3"]
    assert rank1["identity_ids"] == ["4", "5", "6", "7"]
    assert not set(rank0["cl_uids"]).intersection(rank1["cl_uids"])


def test_global_negative_loss_uses_rank_offset_targets() -> None:
    z_all = torch.eye(4)
    # Simulate rank 1 owning rows 2 and 3, with all four global keys gathered.
    z_local = z_all[2:4]
    good = rep.contrastive_loss_with_global_negatives(
        z_local,
        z_local,
        z_all,
        z_all,
        rank=1,
        temperature=0.07,
    )
    bad = rep.contrastive_loss_with_global_negatives(
        z_local,
        torch.flip(z_local, dims=[0]),
        z_all,
        z_all,
        rank=1,
        temperature=0.07,
    )
    assert float(good) < 1e-4
    assert float(bad) > float(good) + 3.0


def test_disable_inplace_activations_preserves_modules_but_turns_flags_off() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.SiLU(inplace=True), nn.ReLU(inplace=True))
    rep.disable_inplace_activations(model)
    assert model[1].inplace is False
    assert model[2].inplace is False


def test_aggregate_seed_summary_records_original_gain() -> None:
    df = pd.DataFrame(
        {
            "condition": ["R1_projection_only"] * 3,
            "seed": [13, 29, 47],
            "best_epoch": [5, 6, 4],
            "duration_seconds": [10, 11, 12],
            "inner_dev_clcb_auc": [0.55, 0.56, 0.54],
            "inner_dev_clcb_eer": [0.45, 0.44, 0.46],
            "mean_cross_modal_mrr": [0.12, 0.13, 0.11],
            "cl_to_cb_recall_at_1": [0.1, 0.1, 0.1],
            "cl_to_cb_recall_at_5": [0.2, 0.2, 0.2],
            "cl_to_cb_mrr": [0.12, 0.13, 0.11],
            "cb_to_cl_recall_at_1": [0.1, 0.1, 0.1],
            "cb_to_cl_recall_at_5": [0.2, 0.2, 0.2],
            "cb_to_cl_mrr": [0.12, 0.13, 0.11],
            "clcl_same_auc": [0.8, 0.8, 0.8],
            "clcl_cross_auc": [0.7, 0.7, 0.7],
            "cbcb_same_auc": [0.8, 0.8, 0.8],
            "cbcb_cross_auc": [0.7, 0.7, 0.7],
            "within_mean_auc": [0.75, 0.75, 0.75],
        }
    )
    summary = rep.aggregate_seed_summary(df)
    row = summary.iloc[0]
    assert row["inner_dev_clcb_auc_median"] == 0.55
    assert np.isclose(row["median_auc_minus_original"], 0.002)
    assert row["seeds_improved_by_at_least_0.05"] == 0


def test_classification_b_when_median_gain_small_and_retrieval_flat() -> None:
    rows = []
    for condition, auc in {
        "R1_projection_only": 0.552,
        "R3_full_shared_encoder_adaptation": 0.560,
    }.items():
        for seed in (13, 29, 47):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": 3,
                    "duration_seconds": 10,
                    "inner_dev_clcb_auc": auc,
                    "inner_dev_clcb_eer": 0.45,
                    "mean_cross_modal_mrr": 0.12,
                    "cl_to_cb_recall_at_1": 0.03,
                    "cl_to_cb_recall_at_5": 0.12,
                    "cl_to_cb_mrr": 0.12,
                    "cb_to_cl_recall_at_1": 0.03,
                    "cb_to_cl_recall_at_5": 0.12,
                    "cb_to_cl_mrr": 0.12,
                    "clcl_same_auc": 0.8,
                    "clcl_cross_auc": 0.7,
                    "cbcb_same_auc": 0.8,
                    "cbcb_cross_auc": 0.7,
                    "within_mean_auc": 0.75,
                }
            )
    seed_metrics = pd.DataFrame(rows)
    seed_summary = rep.aggregate_seed_summary(seed_metrics)
    collapse = pd.DataFrame(
        {
            "condition": ["R1_projection_only", "R3_full_shared_encoder_adaptation"],
            "per_dim_std_mean": [0.02, 0.02],
            "near_identical_fraction": [0.0, 0.0],
        }
    )
    decision = rep.classify_compute_adequacy(
        seed_metrics,
        seed_summary,
        collapse,
        reference={
            "inner_dev_auc": rep.ORIGINAL_INNER_DEV_AUC,
            "inner_dev_mean_cross_modal_mrr": {
                "R1_projection_only": 0.12,
                "R3_full_shared_encoder_adaptation": 0.12,
            },
            "inner_dev_within_mean_auc": {
                "R1_projection_only": 0.75,
                "R3_full_shared_encoder_adaptation": 0.75,
            },
        },
    )
    assert decision["classification"] == "B. GLOBAL_ALIGNMENT_FAILURE_CONFIRMED"
    assert decision["official_val_gate"]["opened"] is False


def test_classification_c_for_isolated_unstable_high_seed() -> None:
    rows = []
    for seed, auc in zip((13, 29, 47), (0.61, 0.54, 0.55)):
        rows.append(
            {
                "condition": "R1_projection_only",
                "seed": seed,
                "best_epoch": 3,
                "duration_seconds": 10,
                "inner_dev_clcb_auc": auc,
                "inner_dev_clcb_eer": 0.45,
                "mean_cross_modal_mrr": 0.12,
                "cl_to_cb_recall_at_1": 0.03,
                "cl_to_cb_recall_at_5": 0.12,
                "cl_to_cb_mrr": 0.12,
                "cb_to_cl_recall_at_1": 0.03,
                "cb_to_cl_recall_at_5": 0.12,
                "cb_to_cl_mrr": 0.12,
                "clcl_same_auc": 0.8,
                "clcl_cross_auc": 0.7,
                "cbcb_same_auc": 0.8,
                "cbcb_cross_auc": 0.7,
                "within_mean_auc": 0.75,
            }
        )
    # Add flat R3 so A is not reachable globally.
    for seed in (13, 29, 47):
        rows.append({**rows[-1], "condition": "R3_full_shared_encoder_adaptation", "seed": seed, "inner_dev_clcb_auc": 0.554})
    seed_metrics = pd.DataFrame(rows)
    seed_summary = rep.aggregate_seed_summary(seed_metrics)
    collapse = pd.DataFrame(
        {
            "condition": ["R1_projection_only", "R3_full_shared_encoder_adaptation"],
            "per_dim_std_mean": [0.02, 0.02],
            "near_identical_fraction": [0.0, 0.0],
        }
    )
    decision = rep.classify_compute_adequacy(
        seed_metrics,
        seed_summary,
        collapse,
        reference={"inner_dev_auc": rep.ORIGINAL_INNER_DEV_AUC, "inner_dev_mean_cross_modal_mrr": {}, "inner_dev_within_mean_auc": {}},
    )
    assert decision["classification"] == "C. OPTIMIZATION_INSTABILITY"


def test_final_gate_round_trip_for_r1_and_r3() -> None:
    for condition, epoch in (
        ("R1_projection_only", 5),
        ("R3_full_shared_encoder_adaptation", 8),
    ):
        gate = rep.encode_final_gate(
            opened=True,
            selected_condition=condition,
            selected_epoch=epoch,
            device=torch.device("cpu"),
        )
        decoded = rep.decode_final_gate(gate)
        assert decoded == {
            "opened": True,
            "selected_condition": condition,
            "selected_epoch": epoch,
        }


def test_closed_final_gate_normalizes_payload_to_empty_selection() -> None:
    gate = rep.encode_final_gate(
        opened=False,
        selected_condition="R3_full_shared_encoder_adaptation",
        selected_epoch=9,
        device=torch.device("cpu"),
    )
    decoded = rep.decode_final_gate(gate)
    assert decoded == {
        "opened": False,
        "selected_condition": "",
        "selected_epoch": 0,
    }


def test_open_final_gate_rejects_invalid_selection() -> None:
    try:
        rep.encode_final_gate(
            opened=True,
            selected_condition="",
            selected_epoch=3,
            device=torch.device("cpu"),
        )
    except rep.ComputeReplicationError:
        pass
    else:
        raise AssertionError("Expected ComputeReplicationError for missing selected condition")
