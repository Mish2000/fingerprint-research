"""Tests for the frozen fusion transfer audit logic (Phase 4A.2B-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnostics import audit_polyu_cross_frozen_fusion_transfer as aud


def test_unseen_categorical_is_incompatible():
    schema = {"numeric_features": ["sourceafis_score"], "categorical_features": ["dataset", "frgp"]}
    a = aud.assess_compatibility(
        schema,
        constructible_numeric={"sourceafis_score"},
        polyu_categoricals={"dataset": {"polyu_cross"}, "frgp": {"0"}},
        sd300_categorical_levels={"dataset": {"nist_sd300b", "nist_sd300c"}, "frgp": {"2", "3", "10"}},
    )
    assert a["verdict"] == "SCHEMA_INCOMPATIBLE"
    assert "dataset" in a["unseen_categoricals"] and "frgp" in a["unseen_categoricals"]


def test_missing_numeric_is_incompatible():
    schema = {"numeric_features": ["sourceafis_score", "source_dpi_a"], "categorical_features": []}
    a = aud.assess_compatibility(
        schema, constructible_numeric={"sourceafis_score"}, polyu_categoricals={}, sd300_categorical_levels={}
    )
    assert a["verdict"] == "SCHEMA_INCOMPATIBLE"
    assert "source_dpi_a" in a["missing_numeric"]


def test_fully_constructible_numeric_only_is_compatible():
    schema = {"numeric_features": ["sourceafis_score", "sift_score"], "categorical_features": []}
    a = aud.assess_compatibility(
        schema, constructible_numeric={"sourceafis_score", "sift_score"}, polyu_categoricals={}, sd300_categorical_levels={}
    )
    assert a["verdict"] == "COMPATIBLE"
    assert a["reasons"] == []


def test_feature_shift_stats_and_out_of_range():
    sd300 = pd.DataFrame({"f": np.linspace(0.0, 1.0, 1000)})
    tr = pd.DataFrame({"f": np.linspace(5.0, 6.0, 100)})  # entirely above SD300 range
    va = pd.DataFrame({"f": np.linspace(5.0, 6.0, 50)})
    s = aud.feature_shift_summary(sd300, tr, va, ["f"])
    row = s[s.polyu_split == "train"].iloc[0]
    assert row["constructible"]
    assert row["frac_polyu_above_sd300_p99"] == 1.0
    assert row["standardized_mean_shift"] > 5.0


def test_categorical_shift_unseen_rate():
    sd300 = pd.DataFrame({"dataset": ["nist_sd300b"] * 10})
    tr = pd.DataFrame({"dataset": ["polyu_cross"] * 8})
    va = pd.DataFrame({"dataset": ["polyu_cross"] * 4})
    c = aud.categorical_shift(sd300, tr, va, ["dataset"])
    assert (c["unseen_rate"] == 1.0).all()
