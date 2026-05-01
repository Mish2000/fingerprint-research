from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

BENCHMARK_RUN_META_SCHEMA_VERSION = "v2_benchmark_run_meta"
BENCHMARK_CONFIG_SCHEMA_VERSION = "v2_benchmark_eval_config"
SCORES_SCHEMA_VERSION = "v1_label_score_csv"

SUMMARY_REQUIRED_COLUMNS = [
    "method",
    "split",
    "n_pairs",
    "auc",
    "eer",
    "tar_at_far_1e_2",
    "tar_at_far_1e_3",
    "scores_csv",
    "config_json",
]

RUN_META_REQUIRED_FIELDS = [
    "schema_version",
    "row",
    "scores_csv",
    "roc_png",
    "summary_csv",
    "resolved_data_dir",
    "manifest_path",
    "pairs_path",
    "config",
]


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        if math.isnan(value):  # type: ignore[arg-type]
            return True
    except Exception:
        pass
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def validate_summary_columns(df: pd.DataFrame, *, context: str = "results_summary.csv") -> None:
    missing = [col for col in SUMMARY_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def validate_scores_csv(
    scores_csv: str | Path,
    *,
    expected_n_pairs: int | None = None,
    context: str = "scores CSV",
) -> pd.DataFrame:
    path = Path(scores_csv)
    if not path.exists():
        raise FileNotFoundError(f"{context} not found: {path}")

    df = pd.read_csv(path)
    required = ["label", "score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns {required}; found {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{context} must not be empty")

    labels = set(df["label"].astype(int).unique().tolist())
    if labels - {0, 1}:
        raise ValueError(f"{context} label column must contain only 0/1; found {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError(f"{context} must contain both positive and negative labels")

    score_series = pd.to_numeric(df["score"], errors="coerce")
    if score_series.isnull().any():
        raise ValueError(f"{context} contains non-numeric score values")

    if expected_n_pairs is not None and int(len(df)) != int(expected_n_pairs):
        raise ValueError(
            f"{context} row count mismatch: expected {int(expected_n_pairs)} rows but found {len(df)}"
        )

    return df


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_scalar(left: Any, right: Any, *, field: str, context: str) -> None:
    if _is_missing_scalar(left) and _is_missing_scalar(right):
        return
    lf = _coerce_float(left)
    rf = _coerce_float(right)
    if lf is not None and rf is not None:
        if not math.isclose(lf, rf, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{context} field {field!r} mismatch: {left!r} != {right!r}")
        return
    if str(left) != str(right):
        raise ValueError(f"{context} field {field!r} mismatch: {left!r} != {right!r}")


def validate_run_meta(
    run_meta_path: str | Path,
    *,
    expected_row: Mapping[str, Any],
    expected_scores_csv: str | Path,
    expected_summary_csv: str | Path,
    expected_method: str,
    expected_split: str,
) -> Mapping[str, Any]:
    path = Path(run_meta_path)
    if not path.exists():
        raise FileNotFoundError(f"run meta not found: {path}")

    payload = load_json(path)
    missing = [field for field in RUN_META_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"run meta missing required fields: {missing}")

    if str(payload.get("schema_version")) != BENCHMARK_RUN_META_SCHEMA_VERSION:
        raise ValueError(
            f"run meta schema_version must be {BENCHMARK_RUN_META_SCHEMA_VERSION!r}; "
            f"found {payload.get('schema_version')!r}"
        )

    row = payload.get("row")
    if not isinstance(row, Mapping):
        raise ValueError("run meta field 'row' must be an object")

    for field in [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "scores_csv",
        "config_json",
    ]:
        if field not in row:
            raise ValueError(f"run meta row missing required field: {field}")

    if str(row.get("method")) != str(expected_method):
        raise ValueError(f"run meta row.method mismatch: {row.get('method')!r} != {expected_method!r}")
    if str(row.get("split")) != str(expected_split):
        raise ValueError(f"run meta row.split mismatch: {row.get('split')!r} != {expected_split!r}")

    for field, expected_value in expected_row.items():
        _compare_scalar(row.get(field), expected_value, field=field, context="run meta row")

    if str(payload.get("scores_csv")) != str(expected_scores_csv):
        raise ValueError("run meta scores_csv path mismatch")
    if str(payload.get("summary_csv")) != str(expected_summary_csv):
        raise ValueError("run meta summary_csv path mismatch")

    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("run meta field 'config' must be an object")
    if str(config.get("schema_version")) != BENCHMARK_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"run meta config.schema_version must be {BENCHMARK_CONFIG_SCHEMA_VERSION!r}; "
            f"found {config.get('schema_version')!r}"
        )
    if str(config.get("method")) != str(expected_method):
        raise ValueError("run meta config.method mismatch")
    if str(config.get("split")) != str(expected_split):
        raise ValueError("run meta config.split mismatch")
    if str(config.get("pairs_path")) != str(payload.get("pairs_path")):
        raise ValueError("run meta config.pairs_path must match run meta pairs_path")
    if str(config.get("manifest_path")) != str(payload.get("manifest_path")):
        raise ValueError("run meta config.manifest_path must match run meta manifest_path")
    if str(config.get("resolved_data_dir")) != str(payload.get("resolved_data_dir")):
        raise ValueError("run meta config.resolved_data_dir must match run meta resolved_data_dir")

    config_from_row = json.loads(str(row.get("config_json")))
    if config_from_row != config:
        raise ValueError("run meta config object must match row.config_json exactly")

    return payload
