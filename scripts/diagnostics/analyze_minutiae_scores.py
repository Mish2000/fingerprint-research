from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THRESHOLDS = (0.25, 0.4, 0.4375, 0.5, 0.55)
JSON_NAME = "minutiae_score_diagnostics.json"
MD_NAME = "minutiae_score_diagnostics.md"


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:"):]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _distribution_by_label(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if column not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for label in (0, 1):
        values = _finite_numeric(df.loc[df["label"].astype(int) == label, column])
        out[str(label)] = _stats(values)
    return out


def _threshold_sweep(df: pd.DataFrame) -> list[dict[str, Any]]:
    labels = df["label"].astype(int).to_numpy()
    scores = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    genuine = labels == 1
    impostor = labels == 0
    n_genuine = int(np.sum(genuine))
    n_impostor = int(np.sum(impostor))
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        accepted = scores >= float(threshold)
        tp = int(np.sum(accepted & genuine))
        fp = int(np.sum(accepted & impostor))
        fn = int(np.sum((~accepted) & genuine))
        tn = int(np.sum((~accepted) & impostor))
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tpr": float(tp / n_genuine) if n_genuine else None,
                "fpr": float(fp / n_impostor) if n_impostor else None,
            }
        )
    return rows


def _combine_numeric_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arrays = [_finite_numeric(df[col]) for col in columns if col in df.columns]
    if not arrays:
        return np.zeros(0, dtype=float)
    return np.concatenate(arrays).astype(float, copy=False)


def _bool_rate(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    present = [col for col in columns if col in df.columns]
    if not present:
        return {"available": False, "rate": None, "n": 0}
    values: list[bool] = []
    for col in present:
        series = df[col]
        for value in series:
            if isinstance(value, bool):
                values.append(value)
            else:
                text = str(value).strip().lower()
                if text in {"true", "1", "yes"}:
                    values.append(True)
                elif text in {"false", "0", "no", ""}:
                    values.append(False)
    if not values:
        return {"available": True, "rate": None, "n": 0}
    return {"available": True, "rate": float(np.mean(values)), "n": int(len(values))}


def _split_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for token in (";", "|", ","):
        if token in text:
            return [part.strip() for part in text.split(token) if part.strip()]
    return [text]


def _quality_flag_rates(df: pd.DataFrame) -> dict[str, Any]:
    columns = [col for col in ("extraction_quality_flags_a", "extraction_quality_flags_b") if col in df.columns]
    if not columns:
        return {"available": False, "rates": {}}
    counts: dict[str, int] = {}
    total_templates = 0
    for col in columns:
        for value in df[col]:
            total_templates += 1
            for flag in _split_flags(value):
                counts[flag] = counts.get(flag, 0) + 1
    rates = {
        flag: {"count": int(count), "rate": float(count / max(total_templates, 1))}
        for flag, count in sorted(counts.items())
    }
    return {"available": True, "template_count": int(total_templates), "rates": rates}


def _extreme_rows(df: pd.DataFrame, *, label: int, ascending: bool, limit: int = 10) -> list[dict[str, Any]]:
    subset = df[df["label"].astype(int) == int(label)].copy()
    if subset.empty:
        return []
    subset["score"] = pd.to_numeric(subset["score"], errors="coerce")
    subset = subset.sort_values("score", ascending=ascending).head(limit)
    preferred = [
        "label",
        "score",
        "path_a",
        "path_b",
        "matched_minutiae",
        "tentative_minutiae",
        "minutiae_count_a",
        "minutiae_count_b",
        "endings_a",
        "endings_b",
        "bifurcations_a",
        "bifurcations_b",
        "saturated_by_max_minutiae_a",
        "saturated_by_max_minutiae_b",
        "extraction_quality_flags_a",
        "extraction_quality_flags_b",
    ]
    columns = [col for col in preferred if col in subset.columns]
    rows: list[dict[str, Any]] = []
    for record in subset[columns].to_dict(orient="records"):
        rows.append({key: _json_scalar(value) for key, value in record.items()})
    return rows


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_scores(scores_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    scores = parse_file_uri(scores_path)
    output_dir = parse_file_uri(outdir)
    df = pd.read_csv(scores)
    missing = {"label", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"{scores} missing required columns: {sorted(missing)}")

    label_counts = {
        str(label): int(count)
        for label, count in df["label"].astype(int).value_counts().sort_index().items()
    }

    payload: dict[str, Any] = {
        "schema_version": "v1_minutiae_score_diagnostics",
        "scores_csv": str(scores),
        "n_rows": int(len(df)),
        "label_counts": label_counts,
        "score_distribution_by_label": _distribution_by_label(df, "score"),
        "threshold_sweep": _threshold_sweep(df),
        "minutiae_count_distribution": _stats(
            _combine_numeric_columns(df, ["minutiae_count_a", "minutiae_count_b", "minutiae_a", "minutiae_b"])
        ),
        "endings_distribution": _stats(_combine_numeric_columns(df, ["endings_a", "endings_b", "kept_endings_a", "kept_endings_b"])),
        "bifurcations_distribution": _stats(
            _combine_numeric_columns(df, ["bifurcations_a", "bifurcations_b", "kept_bifurcations_a", "kept_bifurcations_b"])
        ),
        "raw_candidate_endings_distribution": _stats(
            _combine_numeric_columns(df, ["raw_candidate_endings_a", "raw_candidate_endings_b"])
        ),
        "raw_candidate_bifurcations_distribution": _stats(
            _combine_numeric_columns(df, ["raw_candidate_bifurcations_a", "raw_candidate_bifurcations_b"])
        ),
        "saturation_rate": _bool_rate(df, ["saturated_by_max_minutiae_a", "saturated_by_max_minutiae_b"]),
        "quality_flag_rates": _quality_flag_rates(df),
        "top_impostor_scores": _extreme_rows(df, label=0, ascending=False),
        "lowest_genuine_scores": _extreme_rows(df, label=1, ascending=True),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / JSON_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / MD_NAME).write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _stats_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| {name} | {_fmt(stats.get('n'))} | {_fmt(stats.get('mean'))} | {_fmt(stats.get('median'))} | "
        f"{_fmt(stats.get('q05'))} | {_fmt(stats.get('q95'))} | {_fmt(stats.get('min'))} | {_fmt(stats.get('max'))} |"
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Minutiae Score Diagnostics",
        "",
        f"Scores CSV: `{payload['scores_csv']}`",
        f"Rows: {payload['n_rows']}",
        f"Label counts: `{json.dumps(payload['label_counts'], sort_keys=True)}`",
        "",
        "## Score Distribution",
        "",
        "| label | n | mean | median | q05 | q95 | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in payload["score_distribution_by_label"].items():
        lines.append(_stats_row(label, stats))

    lines.extend(
        [
            "",
            "## Threshold Sweep",
            "",
            "| threshold | TPR | FPR | TP | FP | FN | TN |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["threshold_sweep"]:
        lines.append(
            f"| {_fmt(row['threshold'])} | {_fmt(row['tpr'])} | {_fmt(row['fpr'])} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
        )

    lines.extend(
        [
            "",
            "## Extraction Distributions",
            "",
            "| field | n | mean | median | q05 | q95 | min | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _stats_row("minutiae_count", payload["minutiae_count_distribution"]),
            _stats_row("endings", payload["endings_distribution"]),
            _stats_row("bifurcations", payload["bifurcations_distribution"]),
            _stats_row("raw_candidate_endings", payload["raw_candidate_endings_distribution"]),
            _stats_row("raw_candidate_bifurcations", payload["raw_candidate_bifurcations_distribution"]),
            "",
            "## Saturation And Flags",
            "",
            f"Saturation rate: {_fmt(payload['saturation_rate'].get('rate'))}",
            "",
        ]
    )

    flag_rates = payload["quality_flag_rates"]
    if flag_rates.get("available"):
        lines.extend(["| flag | count | rate |", "| --- | ---: | ---: |"])
        for flag, row in flag_rates.get("rates", {}).items():
            lines.append(f"| {flag} | {row['count']} | {_fmt(row['rate'])} |")
    else:
        lines.append("Quality flags were not present in the input CSV.")

    lines.extend(["", "## Top Impostor Scores", ""])
    lines.append("```json")
    lines.append(json.dumps(payload["top_impostor_scores"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.extend(["", "## Lowest Genuine Scores", ""])
    lines.append("```json")
    lines.append(json.dumps(payload["lowest_genuine_scores"], indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze canonical minutiae score CSV diagnostics.")
    parser.add_argument("--scores", required=True, help="Path to scores_minutiae_*.csv")
    parser.add_argument("--outdir", required=True, help="Directory for JSON and Markdown diagnostics")
    args = parser.parse_args(argv)
    analyze_scores(args.scores, args.outdir)
    print(f"Wrote: {parse_file_uri(args.outdir) / JSON_NAME}")
    print(f"Wrote: {parse_file_uri(args.outdir) / MD_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
