"""Tests for the deep-score saturation audit stats/classification (Phase 4A.1C)."""

from __future__ import annotations

import numpy as np

from scripts.diagnostics import audit_polyu_cross_deep_saturation as sat


def _stats(labels, score, logit, protocol="p", split="val"):
    return sat.saturation_stats(
        np.array(labels), np.array(score, dtype=float), np.array(logit, dtype=float),
        protocol=protocol, label_short=protocol, split=split, source="test",
    )


def test_saturation_hides_signal_in_score_not_logit():
    # Genuine logits above impostor logits, all large enough that float64
    # sigmoid rounds to exactly 1.0 (logit >~ 37) -> probability collapses.
    g_logit = np.linspace(45.0, 60.0, 50)
    i_logit = np.linspace(38.0, 44.0, 150)
    logit = np.concatenate([g_logit, i_logit])
    labels = np.concatenate([np.ones(50, int), np.zeros(150, int)])
    score = 1.0 / (1.0 + np.exp(-logit))  # saturates to exactly 1.0

    s = _stats(labels, score, logit)
    # Logit preserves perfect ranking; score collapses to a constant.
    assert s["auc_logit"] > 0.99
    assert s["auc_score"] < 0.6
    assert s["auc_logit_minus_score"] > 0.3
    assert s["frac_score_eq_1"] == 1.0
    assert s["n_unique_score"] == 1
    assert s["n_unique_logit"] > s["n_unique_score"]


def test_no_saturation_score_and_logit_agree():
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 1.5, 200)  # unsaturated range -> spread probabilities
    labels = (logit + rng.normal(0, 0.5, 200) > 0).astype(int)
    score = 1.0 / (1.0 + np.exp(-logit))
    s = _stats(labels, score, logit)
    assert abs(s["auc_score"] - s["auc_logit"]) < 1e-6  # monotone, no ties collapse
    assert s["spearman_score_logit"] > 0.999


def test_classify_A_hidden_signal():
    rows = [
        {"protocol": p, "split": "val", "auc_score": 0.50, "auc_logit": 0.82, "auc_logit_minus_score": 0.32}
        for p in sat.CB_CB_PROTOCOLS
    ]
    out = sat.classify_cbcb(rows)
    assert out["classification"].startswith("A.")


def test_classify_B_true_failure():
    rows = [
        {"protocol": p, "split": "val", "auc_score": 0.50, "auc_logit": 0.505, "auc_logit_minus_score": 0.005}
        for p in sat.CB_CB_PROTOCOLS
    ]
    out = sat.classify_cbcb(rows)
    assert out["classification"].startswith("B.")


def test_classify_C_mixed():
    rows = [
        {"protocol": p, "split": "val", "auc_score": 0.50, "auc_logit": 0.57, "auc_logit_minus_score": 0.07}
        for p in sat.CB_CB_PROTOCOLS
    ]
    out = sat.classify_cbcb(rows)
    assert out["classification"].startswith("C.")
