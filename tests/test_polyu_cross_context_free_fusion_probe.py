"""Tests for the context-free fusion probe CV/selection logic (Phase 4A.2B-2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnostics import run_polyu_cross_context_free_fusion_probe as probe


def test_assign_folds_deterministic_and_balanced():
    fus = [str(i) for i in range(20)]
    a = probe.assign_folds(fus, k=5)
    b = probe.assign_folds(list(reversed(fus)), k=5)
    assert a == b  # deterministic regardless of input order
    sizes = pd.Series(list(a.values())).value_counts()
    assert set(sizes.index) == {0, 1, 2, 3, 4} and sizes.min() == 4 and sizes.max() == 4


def test_fold_masks_subject_disjoint_and_no_span_leak():
    # finger_units 0..9; partitions via round-robin.
    fold_of = probe.assign_folds([str(i) for i in range(10)], k=5)  # {0:0,1:1,...,5:0,6:1,...}
    rows = [
        {"finger_unit_a": "0", "finger_unit_b": "0"},   # both in partition 0 (positive-like)
        {"finger_unit_a": "0", "finger_unit_b": "5"},   # both in partition 0 (0 and 5 -> fold 0)
        {"finger_unit_a": "0", "finger_unit_b": "1"},   # spans partitions 0 and 1
        {"finger_unit_a": "1", "finger_unit_b": "6"},   # both in partition 1
    ]
    t = pd.DataFrame(rows)
    tr, va = probe.fold_masks(t, fold_of, held=0)
    # Held partition 0: val = pairs fully inside 0 -> rows 0,1
    assert va.tolist() == [True, True, False, False]
    # Train = pairs fully outside 0 -> row 3 (both in 1); row 2 spans -> excluded from both
    assert tr.tolist() == [False, False, False, True]
    # Disjoint
    assert not (tr & va).any()


def test_select_prefers_simpler_within_tol():
    summary = pd.DataFrame([
        {"group": "Q0_quality_only", "mean_auc": 0.50, "n_folds_beating_G0": 0},
        {"group": "G0_p2_sourceafis", "mean_auc": 0.80, "n_folds_beating_G0": 0},
        {"group": "G1_p2_plus_deep", "mean_auc": 0.805, "n_folds_beating_G0": 5},
        {"group": "G2_matcher_geometry", "mean_auc": 0.806, "n_folds_beating_G0": 5},
        {"group": "G3_quality_aware", "mean_auc": 0.807, "n_folds_beating_G0": 5},
        {"group": "G4_reliability_interactions", "mean_auc": 0.808, "n_folds_beating_G0": 5},
    ])
    sel = probe.select_group(summary)
    # best=0.808; G0 at 0.80 is within 0.01 -> simplest wins = G0.
    assert sel["selected_group"] == "G0_p2_sourceafis"


def test_select_requires_majority_over_g0():
    summary = pd.DataFrame([
        {"group": "Q0_quality_only", "mean_auc": 0.50, "n_folds_beating_G0": 0},
        {"group": "G0_p2_sourceafis", "mean_auc": 0.70, "n_folds_beating_G0": 0},
        {"group": "G1_p2_plus_deep", "mean_auc": 0.74, "n_folds_beating_G0": 2},  # +0.04 but only 2/5 folds
        {"group": "G2_matcher_geometry", "mean_auc": 0.70, "n_folds_beating_G0": 0},
        {"group": "G3_quality_aware", "mean_auc": 0.70, "n_folds_beating_G0": 0},
        {"group": "G4_reliability_interactions", "mean_auc": 0.70, "n_folds_beating_G0": 0},
    ])
    sel = probe.select_group(summary)
    assert sel["selected_group"] == "G0_p2_sourceafis" and sel["fell_back_to_G0"] is True


def test_shuffled_labels_are_chance():
    rng = np.random.default_rng(0)
    n = 600
    fu = np.repeat(np.arange(30).astype(str), n // 30)  # 30 finger_units
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 0.3, n)  # informative feature
    t = pd.DataFrame({"finger_unit_a": fu, "finger_unit_b": fu, "label": y, "feat": signal})
    fold_of = probe.assign_folds(sorted(set(fu)), k=5)
    groups = {"G0_p2_sourceafis": ["feat"]}
    # True labels -> strong; shuffled -> ~chance.
    cv_true, _ = probe.cv_evaluate(t.rename(columns={"feat": "p2_sourceafis_score"}), fold_of, {"G0_p2_sourceafis": ["p2_sourceafis_score"]})
    assert np.nanmean(cv_true["auc"]) > 0.8
    shuf = t["label"].to_numpy().copy(); rng.shuffle(shuf)
    cv_shuf, _ = probe.cv_evaluate(t.rename(columns={"feat": "p2_sourceafis_score"}), fold_of, {"G0_p2_sourceafis": ["p2_sourceafis_score"]}, labels_override=shuf)
    assert abs(np.nanmean(cv_shuf["auc"]) - 0.5) < 0.12
