from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_replay_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "prove_deep_fusion_v2_statistical_weight_replay.py"
    spec = importlib.util.spec_from_file_location("statistical_replay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def test_manual_replay_matches_sklearn_predict_proba_for_binary_pipeline():
    replay = load_replay_module()
    df = pd.DataFrame(
        {
            "sourceafis_score": [0.5, 1.2, -0.4, 2.1, 0.0, 1.7, -1.0, 0.9],
            "sift_score": [0.2, 0.8, 0.1, 1.1, -0.2, 0.6, -0.7, 0.4],
            "deep_logit": [-0.1, 1.5, -0.8, 2.4, 0.2, 1.0, -1.4, 0.7],
            "dataset": ["nist_sd300b", "nist_sd300b", "nist_sd300c", "nist_sd300c", "nist_sd300b", "nist_sd300c", "nist_sd300b", "nist_sd300c"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    numeric = ["sourceafis_score", "sift_score", "deep_logit"]
    categorical = ["dataset"]
    model = Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                        ("categorical", one_hot_encoder(), categorical),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=13)),
        ]
    )
    model.fit(df[numeric + categorical], y)

    manual, matrix, names = replay._manual_positive_probability(model, df[numeric + categorical])
    sklearn_scores = replay.sklearn_positive_probability(model, df[numeric + categorical])

    assert matrix.shape[0] == len(df)
    assert len(names) == matrix.shape[1]
    assert np.max(np.abs(manual - sklearn_scores)) < 1e-12


def test_coefficient_group_mapping_for_core_feature_names():
    replay = load_replay_module()
    assert replay.feature_group("sourceafis_score") == "sourceafis"
    assert replay.feature_group("sift_inliers") == "sift"
    assert replay.feature_group("deep_logit") == "deep"
    assert replay.feature_group("a_sharpness_laplacian_var") == "quality"
    assert replay.feature_group("dataset") == "metadata"
