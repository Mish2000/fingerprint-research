from __future__ import annotations

import pandas as pd
import pytest

from src.fpbench.universal.deep_fusion_v2 import DeepFusionV2Error, VARIANTS, fit_variant_model


def test_fit_variant_rejects_non_train_rows():
    df = pd.DataFrame({
        "dataset": ["nist_sd300b", "nist_sd300b", "nist_sd300b", "nist_sd300b"],
        "split": ["train", "train", "val", "train"],
        "pair_id": ["0", "1", "2", "3"],
        "label": [0, 1, 0, 1],
        "sourceafis_score": [0.1, 0.9, 0.2, 0.8],
        "sift_score": [0.1, 0.9, 0.2, 0.8],
        "sift_inliers": [1, 10, 2, 11],
        "sift_matches": [5, 20, 6, 21],
        "sift_k1": [100, 100, 100, 100],
        "sift_k2": [100, 100, 100, 100],
        "deep_score": [0.1, 0.9, 0.2, 0.8],
        "deep_logit": [-2, 2, -1, 1],
        "finger_position": ["7", "7", "7", "7"],
        "frgp": ["7", "7", "7", "7"],
    })
    with pytest.raises(DeepFusionV2Error, match="train rows only"):
        fit_variant_model(df, VARIANTS["sourceafis_sift_deep_logit"])
