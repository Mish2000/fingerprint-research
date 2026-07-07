from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.diagnostics import run_polyu_cross_differentiable_matcher_validity as audit
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local


def test_dual_softmax_score_is_finite_symmetric_and_differentiable() -> None:
    a = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    b = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    score_ab, mass_ab = audit.dual_softmax_scores_from_descriptors(a, b, temperature=0.07)
    score_ba, mass_ba = audit.dual_softmax_scores_from_descriptors(b, a, temperature=0.07)

    assert score_ab.shape == (1,)
    assert mass_ab.shape == (1,)
    assert torch.isfinite(score_ab).all()
    assert torch.isfinite(mass_ab).all()
    assert torch.allclose(score_ab, score_ba, atol=1e-6)
    assert torch.allclose(mass_ab, mass_ba, atol=1e-6)
    assert float(score_ab.item()) > 0.99

    score_ab.sum().backward()
    assert a.grad is not None
    assert float(a.grad.abs().sum()) > 0.0


def test_sinkhorn_partial_assignment_is_finite_symmetric_and_differentiable() -> None:
    a = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    b = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    c = F.normalize(-torch.eye(4, dtype=torch.float32).reshape(1, 4, 4), p=2, dim=2)

    score_ab, mass_ab = audit.sinkhorn_partial_assignment_scores_from_descriptors(
        a,
        b,
        entropy_regularization=0.07,
        iterations=20,
        dustbin_score=0.0,
        dustbin_mass=1.0,
    )
    score_ba, mass_ba = audit.sinkhorn_partial_assignment_scores_from_descriptors(
        b,
        a,
        entropy_regularization=0.07,
        iterations=20,
        dustbin_score=0.0,
        dustbin_mass=1.0,
    )
    score_ac, _mass_ac = audit.sinkhorn_partial_assignment_scores_from_descriptors(
        a,
        c,
        entropy_regularization=0.07,
        iterations=20,
        dustbin_score=0.0,
        dustbin_mass=1.0,
    )

    assert torch.isfinite(score_ab).all()
    assert torch.isfinite(mass_ab).all()
    assert torch.allclose(score_ab, score_ba, atol=1e-5)
    assert torch.allclose(mass_ab, mass_ba, atol=1e-5)
    assert float(score_ab.item()) > float(score_ac.item())

    score_ab.sum().backward()
    assert a.grad is not None
    assert float(a.grad.abs().sum()) > 0.0


def test_score_pair_frame_supports_all_audit_scorers_on_toy_descriptors() -> None:
    grid = np.eye(4, dtype=np.float32).reshape(2, 2, 4)
    cache = local.DescriptorCache(
        sample_uids=["a", "b"],
        uid_to_index={"a": 0, "b": 1},
        local_grids={"a": grid, "b": grid.copy()},
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
    cfg = audit.MatcherAuditConfig(score_batch_size=1)

    for method in audit.ALL_SCORERS:
        scores, details, elapsed = audit.score_pair_frame(
            method=method,
            df=df,
            tensors=tensors,
            device=torch.device("cpu"),
            batch_size=1,
            cfg=cfg,
        )
        assert scores.shape == (1,)
        assert np.isfinite(scores).all()
        assert float(scores[0]) > 0.99
        assert elapsed >= 0.0
        assert details.iloc[0]["method"] == method
        assert not bool(details.iloc[0]["failed"])


def _metric_row(method: str, protocol: str, auc: float, *, runtime: float = 1.0) -> dict[str, object]:
    return {
        "method": method,
        "stage": "inner_dev",
        "protocol": protocol,
        "pair_count": 100,
        "genuine_count": 25,
        "impostor_count": 75,
        "scored_count": 100,
        "failure_count": 0,
        "failed_count": 0,
        "roc_auc": auc,
        "eer": 0.1,
        "genuine_score_mean": 0.7,
        "genuine_score_std": 0.1,
        "genuine_score_median": 0.7,
        "impostor_score_mean": 0.5,
        "impostor_score_std": 0.1,
        "impostor_score_median": 0.5,
        "elapsed_seconds": 0.1,
        "runtime_ms_per_pair": runtime,
        "score_min": 0.1,
        "score_max": 0.9,
        "score_range": 0.8,
        "unique_score_count": 100,
        "unique_score_count_rounded_6": 100,
    }


def _metrics_for_decision(s2_same_cb_auc: float, s3_same_cb_auc: float = 0.55) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocols = [
        audit.CLCB,
        audit.CLCL_SAME,
        audit.CLCL_CROSS,
        audit.CBCB_SAME,
        audit.CBCB_CROSS,
    ]
    rows: list[dict[str, object]] = []
    s0_auc = {
        audit.CLCB: 0.51,
        audit.CLCL_SAME: 0.97,
        audit.CLCL_CROSS: 0.78,
        audit.CBCB_SAME: 0.92,
        audit.CBCB_CROSS: 0.74,
    }
    s2_auc = {
        audit.CLCB: 0.50,
        audit.CLCL_SAME: 0.93,
        audit.CLCL_CROSS: 0.70,
        audit.CBCB_SAME: s2_same_cb_auc,
        audit.CBCB_CROSS: 0.66,
    }
    s3_auc = {
        audit.CLCB: 0.50,
        audit.CLCL_SAME: 0.70,
        audit.CLCL_CROSS: 0.61,
        audit.CBCB_SAME: s3_same_cb_auc,
        audit.CBCB_CROSS: 0.61,
    }
    s1_auc = {
        audit.CLCB: 0.49,
        audit.CLCL_SAME: 0.90,
        audit.CLCL_CROSS: 0.69,
        audit.CBCB_SAME: 0.52,
        audit.CBCB_CROSS: 0.51,
    }
    for method, auc_by_protocol in (
        (audit.S0_HARD_MNN_REFERENCE, s0_auc),
        (audit.S1_CURRENT_TOPK_MAX, s1_auc),
        (audit.S2_DUAL_SOFTMAX, s2_auc),
        (audit.S3_SINKHORN_PARTIAL_ASSIGNMENT, s3_auc),
    ):
        for protocol in protocols:
            rows.append(_metric_row(method, protocol, auc_by_protocol[protocol]))
    runtime = pd.DataFrame(
        {
            "method": [audit.S1_CURRENT_TOPK_MAX, audit.S2_DUAL_SOFTMAX, audit.S3_SINKHORN_PARTIAL_ASSIGNMENT],
            "stage": ["inner_dev", "inner_dev", "inner_dev"],
            "protocol": [audit.CLCB, audit.CLCB, audit.CLCB],
            "pair_count": [100, 100, 100],
            "elapsed_seconds": [0.1, 0.1, 0.1],
            "runtime_ms_per_pair": [1.0, 1.0, 1.0],
        }
    )
    return pd.DataFrame(rows), runtime


def test_matcher_validity_decision_prefers_dual_softmax_when_it_passes() -> None:
    metrics, runtime = _metrics_for_decision(s2_same_cb_auc=0.85)
    decision = audit.matcher_validity_decision(metrics, runtime, cfg=audit.MatcherAuditConfig(), smoke=True)
    assert decision["classification"] == "A. DUAL_SOFTMAX_VALID"
    assert audit.S2_DUAL_SOFTMAX in decision["primary_eligible_matchers"]


def test_matcher_validity_decision_reports_no_surrogate_when_s2_s3_fail() -> None:
    metrics, runtime = _metrics_for_decision(s2_same_cb_auc=0.60, s3_same_cb_auc=0.60)
    decision = audit.matcher_validity_decision(metrics, runtime, cfg=audit.MatcherAuditConfig(), smoke=True)
    assert decision["classification"] in {
        "C. NO_VALID_DIFFERENTIABLE_SURROGATE",
        "D. MIXED_OR_INCONCLUSIVE",
    }
    assert not decision["primary_eligible_matchers"]


def test_audit_runner_source_does_not_instantiate_optimizer() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "Adam" not in source
