from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_professor_1000_pos_neg"
)
DEFAULT_DIAG_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
DEFAULT_V2_SCORE_DIR = DEFAULT_DIAG_DIR / "sift_plain_roll_v2_scores"
DEFAULT_V2_COMMAND_LOG = DEFAULT_V2_SCORE_DIR / "command_log.txt"
DEFAULT_PROFESSOR_THRESHOLD_SCORES = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_professor_1to1_five_methods_far_frr"
    / "scores_sift_val.csv"
)
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
PAIR_SETS = ("positive_1000", "negative_1000")
METHOD_SEMANTICS_EPOCHS = {
    "sift": "sift_runtime_aligned_v1",
    "sift_plain_roll_v2": "sift_plain_roll_v2_research_v1",
}


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _load_pair_set_scores(source_dir: Path, method: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pair_set in PAIR_SETS:
        path = source_dir / f"scores_{method}_{pair_set}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing score CSV: {path}")
        df = pd.read_csv(path)
        missing = {"label", "split", "path_a", "path_b", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["pair_set"] = pair_set
        df["source_scores_csv"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _run_meta_path(source_dir: Path, method: str, pair_set: str) -> Path:
    return source_dir / f"run_{method}_{pair_set}.meta.json"


def _load_run_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _official_pipeline_status(meta: dict[str, Any], method: str, pair_set: str) -> tuple[bool, str]:
    config = meta.get("config", {}) if isinstance(meta, dict) else {}
    row = meta.get("row", {}) if isinstance(meta, dict) else {}
    checks = {
        "run_meta_schema": meta.get("schema_version") == "v2_benchmark_run_meta",
        "method": config.get("method") == method,
        "pair_set": config.get("pair_set_name") == pair_set or row.get("pair_set_name") == pair_set,
        "custom_pairs_file": bool(config.get("custom_pairs_file")),
    }
    ok = all(checks.values())
    details = "; ".join(f"{name}={value}" for name, value in checks.items())
    if ok:
        return True, f"official evaluate.py run meta present ({details})"
    return False, f"official run meta incomplete ({details})"


def _method_source_rows(source_dir: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_set in PAIR_SETS:
        score_path = source_dir / f"scores_{method}_{pair_set}.csv"
        meta_path = _run_meta_path(source_dir, method, pair_set)
        meta = _load_run_meta(meta_path)
        official, details = _official_pipeline_status(meta, method, pair_set)
        config = meta.get("config", {}) if isinstance(meta, dict) else {}
        rows.append(
            {
                "method": method,
                "pair_set": pair_set,
                "score_csv": str(score_path),
                "run_meta_json": str(meta_path),
                "official_pipeline": bool(official),
                "official_pipeline_evidence": details,
                "method_semantics_epoch": (
                    config.get("method_semantics_epoch")
                    or METHOD_SEMANTICS_EPOCHS.get(method, "")
                ),
            }
        )
    return rows


def _method_source_summary(source_dir: Path, method: str) -> dict[str, Any]:
    rows = _method_source_rows(source_dir, method)
    return {
        "method_semantics_epoch": rows[0]["method_semantics_epoch"] if rows else "",
        "positive_scores_csv": rows[0]["score_csv"] if rows else "",
        "negative_scores_csv": rows[1]["score_csv"] if len(rows) > 1 else "",
        "positive_run_meta_json": rows[0]["run_meta_json"] if rows else "",
        "negative_run_meta_json": rows[1]["run_meta_json"] if len(rows) > 1 else "",
        "official_pipeline": all(bool(row["official_pipeline"]) for row in rows),
        "official_pipeline_evidence": " | ".join(str(row["official_pipeline_evidence"]) for row in rows),
    }


def _load_command_log(command_log: Path) -> list[str]:
    if not command_log.exists():
        return []
    commands: list[str] = []
    for line in command_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("python "):
            commands.append(line)
    return commands


def _sift_variant_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    score = _safe_numeric(df["score"])
    inliers = _safe_numeric(df["inliers"])
    return {
        "current_score": score,
        "inliers": inliers,
    }


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        actual_far = false_accepts / n_negative
        if actual_far <= float(target_far):
            return float(threshold), false_accepts, float(actual_far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = labels == 1
    negatives = labels == 0
    accepted = scores >= float(threshold) if math.isfinite(float(threshold)) else np.zeros_like(scores, dtype=bool)
    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_positive = int(np.sum(positives))
    n_negative = int(np.sum(negatives))
    return {
        "tar": float(ta / n_positive) if n_positive else float("nan"),
        "far": float(fa / n_negative) if n_negative else float("nan"),
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _interpretation(method: str, variant: str, target_far: float, test_tar: float, test_far: float) -> str:
    if method == "sift_plain_roll_v2":
        return (
            "Experimental NIST SD300B plain-vs-roll candidate; "
            f"at target FAR {target_far:.1%}, test TAR={test_tar:.3f} and test FAR={test_far:.3f}."
        )
    if variant == "inliers":
        return (
            "Existing SIFT inlier-count score variant from the canonical SIFT matcher; "
            "included as a strong non-v2 comparator."
        )
    return "Canonical SIFT current score semantics; included to confirm existing behavior remains unchanged."


def _combined_interpretation(method: str, variant: str) -> str:
    if method == "sift_plain_roll_v2":
        return (
            "Professor-facing combined VAL+TEST view for direct comparison with the earlier professor update; "
            "official TEST-only rows remain the stricter research evaluation."
        )
    if variant == "inliers":
        return (
            "SIFT inlier-count comparator over all selected professor rows, using a val-calibrated operating threshold."
        )
    return (
        "Canonical SIFT current score over all selected professor rows, aligned to the earlier professor-update "
        "val-calibrated operating threshold."
    )


def _build_rows(
    *,
    method: str,
    scores_df: pd.DataFrame,
    variant_scores: dict[str, np.ndarray],
    target_fars: tuple[float, ...],
) -> list[dict[str, Any]]:
    labels = pd.to_numeric(scores_df["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    splits = scores_df["split"].astype(str).str.strip().str.lower().to_numpy()
    val_mask = splits == "val"
    test_mask = splits == "test"
    rows: list[dict[str, Any]] = []
    for variant, values in variant_scores.items():
        values = np.asarray(values, dtype=float)
        val_labels = labels[val_mask]
        val_scores = values[val_mask]
        test_labels = labels[test_mask]
        test_scores = values[test_mask]
        val_negative_scores = val_scores[val_labels == 0]
        for target_far in target_fars:
            threshold, calibration_fa, calibration_far = _threshold_for_far(val_negative_scores, float(target_far))
            val = _confusion(val_labels, val_scores, threshold)
            test = _confusion(test_labels, test_scores, threshold)
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "calibration_false_accepts": int(calibration_fa),
                    "calibration_far": float(calibration_far),
                    "val_far": float(val["far"]),
                    "val_tar": float(val["tar"]),
                    "test_far": float(test["far"]),
                    "test_tar": float(test["tar"]),
                    "test_ta": int(test["ta"]),
                    "test_fr": int(test["fr"]),
                    "test_fa": int(test["fa"]),
                    "test_tr": int(test["tr"]),
                    "n_val_positive": int(val["n_positive"]),
                    "n_val_negative": int(val["n_negative"]),
                    "n_test_positive": int(test["n_positive"]),
                    "n_test_negative": int(test["n_negative"]),
                    "interpretation": _interpretation(
                        method,
                        variant,
                        float(target_far),
                        float(test["tar"]),
                        float(test["far"]),
                    ),
                }
            )
    return rows


def _build_combined_row(
    *,
    method: str,
    variant: str,
    scores_df: pd.DataFrame,
    scores: np.ndarray,
    target_far: float,
    threshold: float,
    calibration_false_accepts: int,
    calibration_far: float,
    threshold_source: str,
) -> dict[str, Any]:
    labels = pd.to_numeric(scores_df["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    combined = _confusion(labels, np.asarray(scores, dtype=float), float(threshold))
    return {
        "report_table": "professor_combined_val_test_1pct",
        "method": method,
        "variant": variant,
        "target_far": float(target_far),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "calibration_false_accepts": int(calibration_false_accepts),
        "calibration_far": float(calibration_far),
        "combined_far": float(combined["far"]),
        "combined_tar": float(combined["tar"]),
        "combined_ta": int(combined["ta"]),
        "combined_fr": int(combined["fr"]),
        "combined_fa": int(combined["fa"]),
        "combined_tr": int(combined["tr"]),
        "n_combined_positive": int(combined["n_positive"]),
        "n_combined_negative": int(combined["n_negative"]),
        "evaluation_scope": "combined original VAL+TEST selected positive_1000 and negative_1000 rows",
        "interpretation": _combined_interpretation(method, variant),
    }


def build_professor_combined_comparison(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    v2_score_dir: str | Path = DEFAULT_V2_SCORE_DIR,
    *,
    target_far: float = 0.01,
    professor_threshold_scores: str | Path = DEFAULT_PROFESSOR_THRESHOLD_SCORES,
) -> pd.DataFrame:
    benchmark = parse_file_uri(benchmark_dir)
    v2_scores = parse_file_uri(v2_score_dir)
    sift_threshold_scores = parse_file_uri(professor_threshold_scores)

    sift = _load_pair_set_scores(benchmark, "sift")
    sift_v2 = _load_pair_set_scores(v2_scores, "sift_plain_roll_v2")
    sift_calibration = pd.read_csv(sift_threshold_scores)
    sift_calibration_labels = (
        pd.to_numeric(sift_calibration["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    )

    rows: list[dict[str, Any]] = []
    for variant, selected_values, calibration_values in (
        (
            "current_score",
            _safe_numeric(sift["score"]),
            _safe_numeric(sift_calibration["score"]),
        ),
        (
            "inliers",
            _safe_numeric(sift["inliers"]),
            _safe_numeric(sift_calibration["inliers"]),
        ),
    ):
        calibration_negatives = calibration_values[sift_calibration_labels == 0]
        threshold, calibration_fa, calibration_far = _threshold_for_far(calibration_negatives, float(target_far))
        rows.append(
            _build_combined_row(
                method="sift",
                variant=variant,
                scores_df=sift,
                scores=selected_values,
                target_far=float(target_far),
                threshold=threshold,
                calibration_false_accepts=calibration_fa,
                calibration_far=calibration_far,
                threshold_source=f"{sift_threshold_scores} | target FAR <= {float(target_far):.2%}",
            )
        )

    v2_labels = pd.to_numeric(sift_v2["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    v2_splits = sift_v2["split"].astype(str).str.strip().str.lower().to_numpy()
    v2_values = _safe_numeric(sift_v2["score"])
    v2_calibration_negatives = v2_values[(v2_splits == "val") & (v2_labels == 0)]
    threshold, calibration_fa, calibration_far = _threshold_for_far(v2_calibration_negatives, float(target_far))
    rows.append(
        _build_combined_row(
            method="sift_plain_roll_v2",
            variant="official_score",
            scores_df=sift_v2,
            scores=v2_values,
            target_far=float(target_far),
            threshold=threshold,
            calibration_false_accepts=calibration_fa,
            calibration_far=calibration_far,
            threshold_source=(
                f"{v2_scores / 'scores_sift_plain_roll_v2_negative_1000.csv'} original VAL negatives "
                f"| target FAR <= {float(target_far):.2%}"
            ),
        )
    )
    return pd.DataFrame(rows)


def build_comparison(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    v2_score_dir: str | Path = DEFAULT_V2_SCORE_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
    v2_command_log: str | Path = DEFAULT_V2_COMMAND_LOG,
) -> pd.DataFrame:
    benchmark = parse_file_uri(benchmark_dir)
    v2_scores = parse_file_uri(v2_score_dir)
    sift = _load_pair_set_scores(benchmark, "sift")
    sift_v2 = _load_pair_set_scores(v2_scores, "sift_plain_roll_v2")
    rows: list[dict[str, Any]] = []
    rows.extend(_build_rows(method="sift", scores_df=sift, variant_scores=_sift_variant_scores(sift), target_fars=target_fars))
    rows.extend(
        _build_rows(
            method="sift_plain_roll_v2",
            scores_df=sift_v2,
            variant_scores={"official_score": _safe_numeric(sift_v2["score"])},
            target_fars=target_fars,
        )
    )
    report = pd.DataFrame(rows)
    source_summaries = {
        "sift": _method_source_summary(benchmark, "sift"),
        "sift_plain_roll_v2": _method_source_summary(v2_scores, "sift_plain_roll_v2"),
    }
    commands = _load_command_log(parse_file_uri(v2_command_log))
    command_text = " || ".join(commands)
    for method, summary in source_summaries.items():
        mask = report["method"] == method
        for key, value in summary.items():
            report.loc[mask, key] = value
        if method == "sift_plain_roll_v2":
            report.loc[mask, "v2_generation_command_log"] = str(parse_file_uri(v2_command_log))
            report.loc[mask, "v2_generation_commands"] = command_text
        else:
            report.loc[mask, "v2_generation_command_log"] = ""
            report.loc[mask, "v2_generation_commands"] = ""
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def _render_markdown(
    report: pd.DataFrame,
    *,
    benchmark_dir: Path,
    v2_score_dir: Path,
    v2_command_log: Path,
    combined_report: pd.DataFrame,
) -> str:
    provenance_rows = []
    for source_dir, method in ((benchmark_dir, "sift"), (v2_score_dir, "sift_plain_roll_v2")):
        provenance_rows.extend(_method_source_rows(source_dir, method))
    v2_commands = _load_command_log(v2_command_log)
    lines = [
        "# SIFT Plain/Roll v2 Official Comparison",
        "",
        f"Source professor benchmark folder: `{benchmark_dir}`",
        f"SIFT v2 score folder: `{v2_score_dir}`",
        "",
        "Official TEST-only rule: thresholds are calibrated only from original `val` rows, using validation negatives for FAR control, and evaluated only on original `test` rows.",
        "",
        "Professor-facing combined rule: keep the val-calibrated 1% FAR operating threshold, but report metrics over all selected `positive_1000` and `negative_1000` rows combined. Combined VAL+TEST is included only for direct comparison with the earlier professor update; TEST-only is the stricter research evaluation.",
        "",
        "## Provenance",
        "",
        "Official pipeline means the score CSV has paired `v2_benchmark_run_meta` emitted by `pipelines/benchmark/evaluate.py`, with the requested method and selected-pair file recorded in run metadata.",
        "",
        "| method | pair set | score CSV | official pipeline | method_semantics_epoch | run meta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in provenance_rows:
        lines.append(
            f"| {row['method']} | {row['pair_set']} | `{row['score_csv']}` | "
            f"{'yes' if row['official_pipeline'] else 'no'} | `{row['method_semantics_epoch']}` | "
            f"`{row['run_meta_json']}` |"
        )
    lines.extend(
        [
            "",
            "Exact v2 score generation commands:",
            "",
            "```powershell",
            "$env:FPRJ_ROOT='C:\\fingerprint-research'",
        ]
    )
    if v2_commands:
        lines.extend(v2_commands)
    else:
        lines.append(f"# No command entries found in {v2_command_log}")
    lines.extend(
        [
            "```",
            "",
            f"Official command log: `{v2_command_log}`",
        "",
        "Important wording:",
        "- This does not prove that plain-vs-roll is solved.",
        "- `sift_plain_roll_v2` is experimental/research, not canonical.",
        "- It significantly improves TAR at low FAR on the selected professor 1000 val/test protocol.",
        "- The combined VAL+TEST table is for direct comparison with the earlier professor update.",
        "- The official TEST-only table is the stricter research evaluation.",
        "- Minutiae remains weak and should be treated separately because its failures are dominated by dense_skeleton/extraction issues.",
        "- Runtime numbers from focused experiments should not be over-interpreted because caching may affect them.",
        "",
        "## Professor-Facing Combined VAL+TEST 1% FAR",
        "",
        "This table reports the selected `positive_1000` and `negative_1000` rows combined, so each method is summarized over 1000 positives and 1000 negatives. It is meant to line up with the earlier professor-facing update, not to replace the stricter TEST-only evaluation below.",
        "",
        "| method | variant | threshold | TA | FA | TAR | FAR | threshold source | interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    combined_ordered = combined_report.sort_values(["method", "variant"]).reset_index(drop=True)
    for _, row in combined_ordered.iterrows():
        ta = f"{int(row['combined_ta'])}/{int(row['n_combined_positive'])}"
        fa = f"{int(row['combined_fa'])}/{int(row['n_combined_negative'])}"
        lines.append(
            f"| {row['method']} | {row['variant']} | {_fmt(row['threshold'], 6)} | {ta} | {fa} | "
            f"{_fmt_pct(row['combined_tar'])} | {_fmt_pct(row['combined_far'])} | "
            f"`{row['threshold_source']}` | {row['interpretation']} |"
        )
    lines.extend(
        [
        "",
        "## Official Val/Test Operating Points",
        "",
        "| method | variant | target FAR | threshold | val FAR | val TAR | test FAR | test TAR | test TA | test FR | test FA | test TR | n val pos | n val neg | n test pos | n test neg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ordered = report.sort_values(["method", "variant", "target_far"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['method']} | {row['variant']} | {_fmt(row['target_far'])} | {_fmt(row['threshold'], 6)} | "
            f"{_fmt(row['val_far'])} | {_fmt(row['val_tar'])} | {_fmt(row['test_far'])} | "
            f"{_fmt(row['test_tar'])} | {int(row['test_ta'])} | {int(row['test_fr'])} | "
            f"{int(row['test_fa'])} | {int(row['test_tr'])} | {int(row['n_val_positive'])} | "
            f"{int(row['n_val_negative'])} | {int(row['n_test_positive'])} | {int(row['n_test_negative'])} |"
        )
    lines.extend(
        [
            "",
            "## TEST-Only 1% FAR Summary",
            "",
            "This is the stricter research view: thresholds are calibrated on original VAL rows and metrics are reported only on original TEST rows.",
            "",
            "| method | variant | threshold | val FAR | val TAR | test FAR | test TAR | test TA/FR/FA/TR | interpretation |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    one_pct = report[np.isclose(report["target_far"].astype(float), 0.01)].copy()
    one_pct = one_pct.sort_values(["test_tar", "test_far"], ascending=[False, True])
    for _, row in one_pct.iterrows():
        counts = f"{int(row['test_ta'])}/{int(row['test_fr'])}/{int(row['test_fa'])}/{int(row['test_tr'])}"
        lines.append(
            f"| {row['method']} | {row['variant']} | {_fmt(row['threshold'], 6)} | "
            f"{_fmt(row['val_far'])} | {_fmt(row['val_tar'])} | {_fmt(row['test_far'])} | "
            f"{_fmt(row['test_tar'])} | {counts} | {row['interpretation']} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    v2_score_dir: str | Path = DEFAULT_V2_SCORE_DIR,
    outdir: str | Path = DEFAULT_DIAG_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
    v2_command_log: str | Path = DEFAULT_V2_COMMAND_LOG,
    professor_threshold_scores: str | Path = DEFAULT_PROFESSOR_THRESHOLD_SCORES,
) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    v2_scores = parse_file_uri(v2_score_dir)
    command_log = parse_file_uri(v2_command_log)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_comparison(
        benchmark,
        v2_scores,
        target_fars=tuple(float(x) for x in target_fars),
        v2_command_log=command_log,
    )
    combined_report = build_professor_combined_comparison(
        benchmark,
        v2_scores,
        target_far=0.01,
        professor_threshold_scores=professor_threshold_scores,
    )
    csv_path = output / "sift_plain_roll_v2_official_comparison.csv"
    md_path = output / "sift_plain_roll_v2_official_comparison.md"
    csv_report = pd.concat(
        [
            report.assign(report_table="official_val_test_operating_points"),
            combined_report,
        ],
        ignore_index=True,
        sort=False,
    )
    csv_report.to_csv(csv_path, index=False)
    md_path.write_text(
        _render_markdown(
            report,
            benchmark_dir=benchmark,
            v2_score_dir=v2_scores,
            v2_command_log=command_log,
            combined_report=combined_report,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official val/test SIFT vs SIFT Plain/Roll v2 comparison.")
    parser.add_argument("--benchmark_dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--v2_score_dir", default=str(DEFAULT_V2_SCORE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_DIAG_DIR))
    parser.add_argument("--v2_command_log", default=str(DEFAULT_V2_COMMAND_LOG))
    parser.add_argument("--professor_threshold_scores", default=str(DEFAULT_PROFESSOR_THRESHOLD_SCORES))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_outputs(
        args.benchmark_dir,
        args.v2_score_dir,
        args.outdir,
        target_fars=tuple(float(x) for x in args.target_far),
        v2_command_log=args.v2_command_log,
        professor_threshold_scores=args.professor_threshold_scores,
    )
    print("Wrote SIFT Plain/Roll v2 official comparison:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
