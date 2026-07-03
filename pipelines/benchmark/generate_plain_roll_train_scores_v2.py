from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from scripts.diagnostics import run_sourceafis_plain_roll_benchmark as sourceafis
from src.fpbench.universal.pair_bundle_metadata import (
    SD300_RUN_PAIR_BUNDLE_VERSION,
    build_pair_bundle_metadata,
    file_sha256,
)


OUTPUT_SCHEMA_VERSION = "plain_roll_train_scores_v2_anatomical_full_pairs"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_METHODS = ("sourceafis_open", "sift_plain_roll_v2", "sift")
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_train_scores_v2_anatomical_full_pairs"
)
DEFAULT_SOURCEAFIS_TEMPLATE_CACHE_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_sourceafis_v2_anatomical_full_pairs"
    / "run_meta"
    / "sourceafis_raw"
    / "template_cache"
)
EXPECTED_TRAIN = {"rows": 28052, "pos": 7013, "neg": 21039}
SOURCEAFIS_METHOD = "sourceafis_open"
SIFT_METHODS = {"sift_plain_roll_v2", "sift"}
PAIR_KEY = ["dataset", "split", "pair_id"]


class TrainScoreGenerationError(RuntimeError):
    """Raised when train score generation or validation cannot continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_file_uri(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _canonical_pairs_path(dataset: str, *, repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "manifests" / dataset / "pairs_train.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / "pairs_train.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise TrainScoreGenerationError(f"Missing canonical train pair CSV for {dataset}: {candidates}")


def _score_path(outdir: Path, dataset: str, method: str) -> Path:
    return outdir / "scores" / f"scores_{dataset}_{method}_train.csv"


def _score_meta_path(score_csv: Path) -> Path:
    return Path(str(score_csv) + ".meta.json")


def _run_meta_path(outdir: Path, dataset: str, method: str) -> Path:
    return outdir / "run_meta" / f"run_{dataset}_{method}_train.meta.json"


def _roc_path(outdir: Path, dataset: str, method: str) -> Path:
    return outdir / "roc" / f"roc_{dataset}_{method}_train.png"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _labels(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)


def _frgp(df: pd.DataFrame) -> pd.Series:
    if "frgp" in df.columns:
        return pd.to_numeric(df["frgp"], errors="coerce").fillna(-1).astype(int)
    if "finger_position" in df.columns:
        return pd.to_numeric(df["finger_position"], errors="coerce").fillna(-1).astype(int)
    raise TrainScoreGenerationError("Missing frgp/finger_position column.")


def _frgp_counts(df: pd.DataFrame) -> dict[str, int]:
    values = _frgp(df)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items() if int(key) > 0}


def _score_stats(df: pd.DataFrame) -> dict[str, Any]:
    labels = _labels(df)
    frgp = _frgp(df)
    rows = {
        "rows": int(len(df)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "frgp_coverage": ",".join(str(value) for value in sorted(set(frgp[frgp > 0].tolist()))),
        "frgp_counts": _frgp_counts(df),
    }
    for value in (1, 6):
        rows[f"frgp_{value}_positive_count"] = int(((frgp == value) & (labels == 1)).sum())
        rows[f"frgp_{value}_negative_count"] = int(((frgp == value) & (labels == 0)).sum())
    return rows


def _image_capture_code(path: Any) -> tuple[str, str]:
    text = Path(str(path)).name.lower()
    parts = text.split("_")
    if len(parts) < 4:
        return "", ""
    capture = parts[-3]
    code = parts[-1].split(".")[0]
    if capture not in {"plain", "roll", "rolled"}:
        return "", ""
    if capture == "rolled":
        capture = "roll"
    return capture, code


def _pair_capture_codes(row: pd.Series) -> tuple[str, str, str, str]:
    cap_a, code_a = _image_capture_code(row["path_a"])
    cap_b, code_b = _image_capture_code(row["path_b"])
    plain_code = code_a if cap_a == "plain" else code_b if cap_b == "plain" else ""
    roll_code = code_a if cap_a == "roll" else code_b if cap_b == "roll" else ""
    return cap_a, code_a, plain_code, roll_code


def _extra_train_pair_audit(dataset: str, pairs: pd.DataFrame, pair_source: Path) -> dict[str, Any]:
    labels = _labels(pairs)
    frgp = _frgp(pairs)
    capture_rows = [_pair_capture_codes(row) for _, row in pairs.iterrows()]
    plain_codes = [plain for _cap, _code, plain, _roll in capture_rows if plain]
    roll_pairs = [(plain, roll) for _cap, _code, plain, roll in capture_rows]
    positive_roll_pairs = [
        (plain, roll)
        for label, (_cap, _code, plain, roll) in zip(labels.tolist(), capture_rows)
        if int(label) == 1
    ]
    checks = {
        "expected_rows": int(len(pairs) == EXPECTED_TRAIN["rows"]),
        "expected_positive_count": int((labels == 1).sum() == EXPECTED_TRAIN["pos"]),
        "expected_negative_count": int((labels == 0).sum() == EXPECTED_TRAIN["neg"]),
        "frgp_coverage_1_to_10": sorted(set(frgp[frgp > 0].tolist())) == list(range(1, 11)),
        "frgp_1_positive_exists": bool(((frgp == 1) & (labels == 1)).any()),
        "frgp_1_negative_exists": bool(((frgp == 1) & (labels == 0)).any()),
        "frgp_6_positive_exists": bool(((frgp == 6) & (labels == 1)).any()),
        "frgp_6_negative_exists": bool(((frgp == 6) & (labels == 0)).any()),
        "plain_13_14_refs_zero": not any(code in {"13", "14"} for code in plain_codes),
        "plain_11_roll_01_positives_exist": ("11", "01") in positive_roll_pairs,
        "plain_12_roll_06_positives_exist": ("12", "06") in positive_roll_pairs,
        "plain_11_roll_11_zero": ("11", "11") not in roll_pairs,
        "plain_12_roll_12_zero": ("12", "12") not in roll_pairs,
    }
    stats = _score_stats(pairs)
    return {
        "dataset": dataset,
        "split": "train",
        "pair_source_path": str(pair_source),
        "pair_source_sha256": file_sha256(pair_source),
        **stats,
        "plain_13_14_ref_count": int(sum(1 for code in plain_codes if code in {"13", "14"})),
        "plain_11_roll_01_positive_count": int(sum(1 for item in positive_roll_pairs if item == ("11", "01"))),
        "plain_12_roll_06_positive_count": int(sum(1 for item in positive_roll_pairs if item == ("12", "06"))),
        "plain_11_roll_11_count": int(sum(1 for item in roll_pairs if item == ("11", "11"))),
        "plain_12_roll_12_count": int(sum(1 for item in roll_pairs if item == ("12", "12"))),
        "required_checks": checks,
        "pass": bool(all(checks.values())),
    }


def _write_train_pair_audits(
    *,
    datasets: tuple[str, ...],
    outdir: Path,
    repo_root: Path,
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]]]:
    selected_dir = outdir / "selected_pairs"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[tuple[str, str], Path] = {}
    pair_sources: dict[tuple[str, str], Path] = {}
    extra_reports: list[dict[str, Any]] = []

    for dataset in datasets:
        src = _canonical_pairs_path(dataset, repo_root=repo_root)
        dst = selected_dir / f"pairs_{dataset}_train.csv"
        shutil.copy2(src, dst)
        selected[(dataset, "train")] = dst
        pair_sources[(dataset, "train")] = src

    audit_dir = outdir / "pair_audit"
    reports = final.write_pair_audits(selected_pairs=selected, pair_audit_out=audit_dir, repo_root=repo_root)
    for report in reports:
        pair_source = pair_sources[(report.dataset, report.split)]
        pairs = pd.read_csv(pair_source)
        extra = _extra_train_pair_audit(report.dataset, pairs, pair_source)
        payload = _read_json(report.json_path)
        payload.update(extra)
        _write_json(report.json_path, payload)
        extra_reports.append(payload)
        report.markdown_path.write_text(_render_pair_audit_markdown(payload), encoding="utf-8")

    _write_pair_audit_summary(extra_reports, audit_dir / "pair_audit_summary.md")
    failed = [item for item in extra_reports if not bool(item.get("pass"))]
    if failed:
        details = "; ".join(f"{item['dataset']}/train" for item in failed)
        raise TrainScoreGenerationError(f"Train pair audit failed before scoring: {details}")
    return selected, extra_reports


def _render_pair_audit_markdown(payload: dict[str, Any]) -> str:
    checks = payload.get("required_checks", {})
    lines = [
        f"# Pair Audit: {payload.get('dataset')} train",
        "",
        f"- pass: `{bool(payload.get('pass'))}`",
        f"- rows: `{payload.get('rows')}`",
        f"- positives: `{payload.get('positive_count')}`",
        f"- negatives: `{payload.get('negative_count')}`",
        f"- frgp coverage: `{payload.get('frgp_coverage')}`",
        f"- pair_source_sha256: `{payload.get('pair_source_sha256')}`",
        "",
        "| check | pass |",
        "| --- | ---: |",
    ]
    for name in sorted(checks):
        lines.append(f"| {name} | {bool(checks[name])} |")
    return "\n".join(lines) + "\n"


def _write_pair_audit_summary(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Train Pair Audit Summary",
        "",
        "| dataset | split | pass | rows | positives | negatives | frgp | frgp1 pos/neg | frgp6 pos/neg | pair sha256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for report in reports:
        lines.append(
            "| {dataset} | train | {passed} | {rows} | {pos} | {neg} | {frgp} | {f1p}/{f1n} | {f6p}/{f6n} | {sha} |".format(
                dataset=report.get("dataset", ""),
                passed=bool(report.get("pass")),
                rows=report.get("rows", ""),
                pos=report.get("positive_count", ""),
                neg=report.get("negative_count", ""),
                frgp=report.get("frgp_coverage", ""),
                f1p=report.get("frgp_1_positive_count", ""),
                f1n=report.get("frgp_1_negative_count", ""),
                f6p=report.get("frgp_6_positive_count", ""),
                f6n=report.get("frgp_6_negative_count", ""),
                sha=report.get("pair_source_sha256", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_alignment_keys(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "split" not in out.columns:
        out["split"] = split
    if "finger_position" in out.columns and "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]
    required = ["pair_id", "label", "path_a", "path_b", "frgp"]
    missing = sorted(column for column in required if column not in out.columns)
    if missing:
        raise TrainScoreGenerationError(f"Alignment table is missing columns {missing}. Found={list(out.columns)}")
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    out["frgp"] = pd.to_numeric(out["frgp"], errors="coerce").fillna(-1).astype(int)
    out["path_a"] = out["path_a"].astype(str)
    out["path_b"] = out["path_b"].astype(str)
    return out[out["split"] == split].copy()


def _attach_pair_traceability(*, dataset: str, pairs_csv: Path, scores_csv: Path, method: str) -> None:
    pairs = pd.read_csv(pairs_csv)
    scores = pd.read_csv(scores_csv)
    if len(pairs) != len(scores):
        raise TrainScoreGenerationError(
            f"Score row count does not match canonical pairs for {scores_csv}: scores={len(scores)} pairs={len(pairs)}"
        )
    updated = scores.copy()
    updated.insert(0, "method", method) if "method" not in updated.columns else None
    updated["dataset"] = dataset
    for column in ("pair_id", "label", "split", "subject_a", "subject_b", "path_a", "path_b", "frgp"):
        if column in pairs.columns:
            updated[column] = pairs[column].to_numpy()
    if "frgp" in pairs.columns:
        updated["finger_position"] = pairs["frgp"].astype(str).to_numpy()
    updated.to_csv(scores_csv, index=False)


def _alignment_digest(df: pd.DataFrame) -> str:
    text = df[PAIR_KEY + ["label", "frgp", "path_a", "path_b"]].sort_values(PAIR_KEY).to_csv(index=False, lineterminator="\n")
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_score_alignment(
    *,
    dataset: str,
    method: str,
    score_csv: Path,
    repo_root: Path,
    require_meta: bool = True,
) -> dict[str, Any]:
    pairs_csv = _canonical_pairs_path(dataset, repo_root=repo_root)
    pairs = _normalize_alignment_keys(pd.read_csv(pairs_csv), dataset=dataset, split="train")
    scores = _normalize_alignment_keys(pd.read_csv(score_csv), dataset=dataset, split="train")
    if len(scores) != len(pairs):
        raise TrainScoreGenerationError(f"{score_csv}: row count mismatch scores={len(scores)} canonical={len(pairs)}")
    if scores.duplicated(PAIR_KEY).any():
        examples = scores.loc[scores.duplicated(PAIR_KEY, keep=False), PAIR_KEY].head(5).to_dict("records")
        raise TrainScoreGenerationError(f"{score_csv}: duplicate pair keys: {examples}")

    expected = pairs[PAIR_KEY + ["label", "frgp", "path_a", "path_b"]].copy()
    actual = scores[PAIR_KEY + ["label", "frgp", "path_a", "path_b"]].copy()
    merged = expected.merge(actual, on=PAIR_KEY, how="outer", suffixes=("_pair", "_score"), indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        raise TrainScoreGenerationError(f"{score_csv}: pair_id set mismatch missing={missing} extra={extra}")
    both = merged[merged["_merge"] == "both"].copy()
    mismatches = {
        "label_mismatch": int((both["label_pair"] != both["label_score"]).sum()),
        "frgp_mismatch": int((both["frgp_pair"] != both["frgp_score"]).sum()),
        "path_a_mismatch": int((both["path_a_pair"].astype(str) != both["path_a_score"].astype(str)).sum()),
        "path_b_mismatch": int((both["path_b_pair"].astype(str) != both["path_b_score"].astype(str)).sum()),
    }
    bad = {key: value for key, value in mismatches.items() if value}
    if bad:
        raise TrainScoreGenerationError(f"{score_csv}: alignment mismatches: {bad}")

    meta_path = _score_meta_path(score_csv)
    meta = _read_json(meta_path)
    pair_meta = build_pair_bundle_metadata(dataset=dataset, split="train", pair_source_path=pairs_csv, repo_root=repo_root)
    if require_meta:
        required = {
            "dataset_id": dataset,
            "split": "train",
            "method": method,
            "pair_source_sha256": pair_meta["pair_source_sha256"],
            "manifest_source_sha256": pair_meta["manifest_source_sha256"],
            "split_subjects_sha256": pair_meta["split_subjects_sha256"],
            "sd300_frgp_semantics": "anatomical",
            "run_pair_bundle_version": SD300_RUN_PAIR_BUNDLE_VERSION,
        }
        for key, expected_value in required.items():
            if str(meta.get(key, "")) != str(expected_value):
                raise TrainScoreGenerationError(
                    f"{score_csv}: meta field {key!r} mismatch expected={expected_value!r} actual={meta.get(key)!r}"
                )
    stats = _score_stats(scores)
    coverage = [int(item) for item in stats["frgp_coverage"].split(",") if item]
    if stats["rows"] != EXPECTED_TRAIN["rows"] or stats["positive_count"] != EXPECTED_TRAIN["pos"] or stats["negative_count"] != EXPECTED_TRAIN["neg"]:
        raise TrainScoreGenerationError(f"{score_csv}: label counts mismatch: {stats}")
    if coverage != list(range(1, 11)):
        raise TrainScoreGenerationError(f"{score_csv}: frgp coverage mismatch: {coverage}")
    for value in (1, 6):
        if int(stats[f"frgp_{value}_positive_count"]) <= 0 or int(stats[f"frgp_{value}_negative_count"]) <= 0:
            raise TrainScoreGenerationError(f"{score_csv}: frgp={value} missing positive or negative rows")
    return {
        "dataset": dataset,
        "method": method,
        "split": "train",
        "score_csv": str(score_csv),
        "meta_json": str(meta_path),
        "alignment_digest": _alignment_digest(scores),
        "pair_source_sha256": pair_meta["pair_source_sha256"],
        "manifest_source_sha256": pair_meta["manifest_source_sha256"],
        **stats,
        "status": "pass",
    }


def _merge_metadata(
    *,
    dataset: str,
    method: str,
    score_csv: Path,
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pairs_csv = _canonical_pairs_path(dataset, repo_root=repo_root)
    pair_meta = build_pair_bundle_metadata(dataset=dataset, split="train", pair_source_path=pairs_csv, repo_root=repo_root)
    scores = pd.read_csv(score_csv)
    stats = _score_stats(scores)
    payload: dict[str, Any] = {
        **pair_meta,
        "method": method,
        "scores_csv": str(score_csv),
        "score_count": int(len(scores)),
        "created_at": _utc_now(),
        "score_summary": stats,
        "cache_policy": "score rows recomputed unless score_table_reused_with_matching_sha is true",
        "score_table_reused_with_matching_sha": False,
    }
    if method == SOURCEAFIS_METHOD:
        payload["sourceafis_raw_reused"] = False
    if extra:
        payload.update(extra)
    meta_path = _score_meta_path(score_csv)
    existing = _read_json(meta_path)
    existing.update(payload)
    existing["pair_bundle_metadata"] = pair_meta
    _write_json(meta_path, existing)
    return existing


def _merge_run_meta(run_meta: Path, updates: dict[str, Any]) -> None:
    if not run_meta.exists():
        return
    payload = _read_json(run_meta)
    payload.update(updates)
    payload["pair_bundle_metadata"] = {
        key: updates[key]
        for key in (
            "dataset_id",
            "split",
            "pair_source_path",
            "pair_source_sha256",
            "manifest_source_path",
            "manifest_source_sha256",
            "split_subjects_path",
            "split_subjects_sha256",
            "pair_count",
            "positive_count",
            "negative_count",
            "frgp_counts",
            "sd300_frgp_semantics",
            "sd300_raw_frgp_available",
            "run_pair_bundle_version",
        )
        if key in updates
    }
    _write_json(run_meta, payload)


def _run_subprocess(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(cwd)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise TrainScoreGenerationError(
            "Command failed with exit code "
            f"{proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)


def _score_sift_method(*, dataset: str, method: str, outdir: Path, repo_root: Path) -> Path:
    pairs_csv = _canonical_pairs_path(dataset, repo_root=repo_root)
    score_csv = _score_path(outdir, dataset, method)
    run_meta = _run_meta_path(outdir, dataset, method)
    roc = _roc_path(outdir, dataset, method)
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    run_meta.parent.mkdir(parents=True, exist_ok=True)
    roc.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "pipelines" / "benchmark" / "evaluate.py"),
        "--method",
        method,
        "--dataset",
        dataset,
        "--split",
        "train",
        "--data_dir",
        str(repo_root / "data" / "manifests" / dataset),
        "--pairs_file",
        str(pairs_csv),
        "--pair_set_name",
        "train",
        "--limit",
        "0",
        "--out_scores",
        str(score_csv),
        "--out_roc",
        str(roc),
        "--out_run_meta",
        str(run_meta),
        "--summary_csv",
        str(outdir / "evaluate_results_summary.csv"),
    ]
    print(f"[score] {dataset} {method} train")
    _run_subprocess(cmd, cwd=repo_root)
    _attach_pair_traceability(dataset=dataset, pairs_csv=pairs_csv, scores_csv=score_csv, method=method)
    updates = _merge_metadata(dataset=dataset, method=method, score_csv=score_csv, repo_root=repo_root)
    _merge_run_meta(run_meta, updates)
    validate_score_alignment(dataset=dataset, method=method, score_csv=score_csv, repo_root=repo_root)
    return score_csv


def _sourceafis_pair_total_ms(scores: pd.DataFrame) -> pd.Series:
    extraction_a = pd.to_numeric(scores.get("extraction_latency_ms_a", 0.0), errors="coerce").fillna(0.0)
    extraction_b = pd.to_numeric(scores.get("extraction_latency_ms_b", 0.0), errors="coerce").fillna(0.0)
    verification_wall = pd.to_numeric(scores.get("verification_wall_latency_ms", np.nan), errors="coerce")
    verification_reported = pd.to_numeric(scores.get("verification_latency_ms", np.nan), errors="coerce")
    verification = verification_wall.where(np.isfinite(verification_wall), verification_reported).fillna(0.0)
    return extraction_a + extraction_b + verification


def _write_sourceafis_score_csv(scored: pd.DataFrame, score_csv: Path) -> None:
    out = scored.copy()
    if "method" not in out.columns:
        out.insert(0, "method", SOURCEAFIS_METHOD)
    out["score"] = pd.to_numeric(out["raw_score"], errors="coerce")
    out["pair_total_ms"] = _sourceafis_pair_total_ms(out)
    if "finger_position" in out.columns and "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]
    columns = [
        "method",
        "dataset",
        "split",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "finger_position",
        "frgp",
        "path_a",
        "path_b",
        "score",
        "raw_score",
        "score_semantics",
        "higher_is_more_similar",
        "dpi_a",
        "dpi_b",
        "provider_id",
        "provider_version",
        "template_format",
        "template_version",
        "extraction_cache_hit_a",
        "extraction_cache_hit_b",
        "extraction_latency_ms_a",
        "extraction_latency_ms_b",
        "verification_latency_ms",
        "verification_wall_latency_ms",
        "pair_total_ms",
        "warnings",
        "error",
    ]
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(score_csv, index=False)


def _score_sourceafis(
    *,
    datasets: tuple[str, ...],
    outdir: Path,
    repo_root: Path,
    template_cache_dir: Path,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
) -> dict[str, Path]:
    print("[score] sourceafis_open train")
    sourceafis.validate_dpi_settings(dpi_strategy=dpi_strategy, image_dpi=image_dpi)
    timeout_overrides = sourceafis._timeout_env_overrides(
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
    )
    previous_env = sourceafis._set_env_overrides(timeout_overrides)
    try:
        engine, provider_metadata = sourceafis.ensure_sourceafis_available(require_enabled_env=True)
        warmup = sourceafis.warmup_sidecar(engine)
        if not bool(warmup.get("ok")):
            raise TrainScoreGenerationError(
                f"SourceAFIS sidecar warmup failed: {warmup.get('error_message') or warmup.get('unavailable_reason')}"
            )
        service_url = str(
            provider_metadata.metadata.get("service_url")
            or os.getenv(sourceafis.SOURCEAFIS_SERVICE_URL_ENV)
            or warmup.get("service_url")
            or ""
        )
        cache = sourceafis.TemplateCache(
            template_cache_dir,
            provider_metadata=provider_metadata,
            service_url=service_url,
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            repo_root=repo_root,
        )
        retry_config = sourceafis.RetryConfig(
            max_retries=max(int(max_retries), 0),
            retry_backoff_seconds=max(float(retry_backoff_seconds), 0.0),
        )
        generated: dict[str, Path] = {}
        all_failures: list[dict[str, Any]] = []
        for dataset in datasets:
            pairs, status = sourceafis.load_plain_roll_pairs(
                dataset,
                "train",
                repo_root=repo_root,
                limit=0,
                sample_strategy=sourceafis.DEFAULT_SAMPLE_STRATEGY,
                sample_seed=sourceafis.DEFAULT_SAMPLE_SEED,
            )
            if pairs.empty:
                raise TrainScoreGenerationError(f"No SourceAFIS-compatible train pairs for {dataset}: {status}")
            sourceafis.validate_dataset_dpi(pairs, dataset=dataset, dpi_strategy=dpi_strategy, image_dpi=image_dpi)
            scored, failures, events = sourceafis.score_pairs(
                pairs,
                engine=engine,
                cache=cache,
                retry_config=retry_config,
            )
            if failures:
                all_failures.extend(failures)
                raise TrainScoreGenerationError(f"SourceAFIS failures while scoring {dataset}: {len(failures)}")
            score_csv = _score_path(outdir, dataset, SOURCEAFIS_METHOD)
            _write_sourceafis_score_csv(scored, score_csv)
            extra = {
                "sourceafis_raw_reused": False,
                "provider_id": provider_metadata.provider_id,
                "provider_version": provider_metadata.provider_version,
                "template_format": provider_metadata.template_format,
                "template_version": provider_metadata.template_version,
                "service_url": service_url,
                "template_cache_dir": str(cache.cache_dir),
                "template_cache_file_count_after_run": int(len(list(cache.cache_dir.glob("*.json")))),
                "sourceafis_latency_event_count": int(len(events)),
            }
            updates = _merge_metadata(
                dataset=dataset,
                method=SOURCEAFIS_METHOD,
                score_csv=score_csv,
                repo_root=repo_root,
                extra=extra,
            )
            run_meta = _run_meta_path(outdir, dataset, SOURCEAFIS_METHOD)
            _write_json(
                run_meta,
                {
                    "schema_version": "sourceafis_open_train_score_run_meta_v1",
                    "method": SOURCEAFIS_METHOD,
                    "dataset": dataset,
                    "split": "train",
                    "scores_csv": str(score_csv),
                    "method_meta_json": str(_score_meta_path(score_csv)),
                    "sourceafis_raw_reused": False,
                    "timing": {
                        "avg_ms_pair_reported": float(pd.to_numeric(scored.get("pair_total_ms", pd.Series(dtype=float)), errors="coerce").mean())
                        if "pair_total_ms" in scored
                        else float("nan"),
                    },
                    **updates,
                },
            )
            validate_score_alignment(dataset=dataset, method=SOURCEAFIS_METHOD, score_csv=score_csv, repo_root=repo_root)
            generated[dataset] = score_csv
        failures_path = outdir / "sourceafis_train_failures.csv"
        pd.DataFrame(all_failures, columns=sourceafis.FAILURE_COLUMNS).to_csv(failures_path, index=False)
        return generated
    finally:
        sourceafis._restore_env(previous_env)


def _write_score_summary(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_existing_artifacts(
    *,
    outdir: Path = DEFAULT_OUTDIR,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    output = parse_file_uri(outdir, repo_root=repo_root)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            score_csv = _score_path(output, dataset, method)
            if not score_csv.exists():
                raise TrainScoreGenerationError(f"Missing score CSV: {score_csv}")
            rows.append(validate_score_alignment(dataset=dataset, method=method, score_csv=score_csv, repo_root=repo_root))
    return rows


def _write_manifest(
    *,
    outdir: Path,
    repo_root: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    pair_audits: list[dict[str, Any]],
    score_summary: list[dict[str, Any]],
    commands: list[list[str]],
    total_runtime_s: float,
    template_cache_dir: Path,
) -> None:
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "datasets": list(datasets),
        "methods": list(methods),
        "split": "train",
        "run_pair_bundle_version": SD300_RUN_PAIR_BUNDLE_VERSION,
        "output_dir": str(outdir),
        "scores_dir": str(outdir / "scores"),
        "selected_pairs_dir": str(outdir / "selected_pairs"),
        "pair_audit_dir": str(outdir / "pair_audit"),
        "pair_audits": pair_audits,
        "score_summary": score_summary,
        "commands": commands,
        "sourceafis_cache_policy": {
            "score_rows": "recomputed for train; no legacy train score CSV copied",
            "template_cache_dir": str(template_cache_dir),
            "template_cache_reuse": "per-image template cache only; score rows are newly written",
        },
        "legacy_train_scores_used": False,
        "forbidden_inputs": [
            "artifacts/reports/**/selected_pairs/*.csv as scoring input",
            "artifacts/reports/benchmark/plain_roll_train_scores_v1/**",
            "artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_train_scores/**",
        ],
        "total_runtime_s": float(total_runtime_s),
    }
    _write_json(outdir / "plain_roll_train_scores_manifest.json", manifest)


def run_generation(
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    outdir: Path,
    repo_root: Path,
    sourceafis_template_cache_dir: Path,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
    skip_sourceafis: bool,
) -> dict[str, Path]:
    start = time.perf_counter()
    output = parse_file_uri(outdir, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    selected, pair_audits = _write_train_pair_audits(datasets=datasets, outdir=output, repo_root=repo_root)
    del selected

    if SOURCEAFIS_METHOD in methods and not skip_sourceafis:
        _score_sourceafis(
            datasets=datasets,
            outdir=output,
            repo_root=repo_root,
            template_cache_dir=parse_file_uri(sourceafis_template_cache_dir, repo_root=repo_root),
            request_timeout_seconds=request_timeout_seconds,
            extract_timeout_seconds=extract_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
        )

    for method in methods:
        if method == SOURCEAFIS_METHOD:
            continue
        if method not in SIFT_METHODS:
            raise TrainScoreGenerationError(f"Unsupported train score method: {method}")
        for dataset in datasets:
            _score_sift_method(dataset=dataset, method=method, outdir=output, repo_root=repo_root)

    score_summary = validate_existing_artifacts(outdir=output, datasets=datasets, methods=methods, repo_root=repo_root)
    _write_score_summary(score_summary, output / "score_summary.csv")
    _write_manifest(
        outdir=output,
        repo_root=repo_root,
        datasets=datasets,
        methods=methods,
        pair_audits=pair_audits,
        score_summary=score_summary,
        commands=commands,
        total_runtime_s=time.perf_counter() - start,
        template_cache_dir=parse_file_uri(sourceafis_template_cache_dir, repo_root=repo_root),
    )
    return {
        "manifest": output / "plain_roll_train_scores_manifest.json",
        "score_summary": output / "score_summary.csv",
        "pair_audit_summary": output / "pair_audit" / "pair_audit_summary.md",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SD300 anatomical full-pair train score tables for fusion inputs.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--sourceafis_template_cache_dir", default=str(DEFAULT_SOURCEAFIS_TEMPLATE_CACHE_DIR))
    parser.add_argument("--request_timeout_seconds", type=float, default=None)
    parser.add_argument("--extract_timeout_seconds", type=float, default=None)
    parser.add_argument("--verify_timeout_seconds", type=float, default=None)
    parser.add_argument("--max_retries", type=int, default=sourceafis.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_backoff_seconds", type=float, default=sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--dpi_strategy", choices=sourceafis.DPI_STRATEGIES, default=sourceafis.DEFAULT_DPI_STRATEGY)
    parser.add_argument("--image_dpi", type=int, default=None)
    parser.add_argument("--skip_sourceafis", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = parse_file_uri(args.repo_root)
    datasets = _parse_csv_arg(args.datasets)
    methods = _parse_csv_arg(args.methods)
    try:
        if args.validate_only:
            rows = validate_existing_artifacts(
                outdir=parse_file_uri(args.outdir, repo_root=repo_root),
                datasets=datasets,
                methods=methods,
                repo_root=repo_root,
            )
            _write_score_summary(rows, parse_file_uri(args.outdir, repo_root=repo_root) / "score_summary.csv")
            print("Train score alignment validation passed.")
            return 0
        paths = run_generation(
            datasets=datasets,
            methods=methods,
            outdir=parse_file_uri(args.outdir, repo_root=repo_root),
            repo_root=repo_root,
            sourceafis_template_cache_dir=parse_file_uri(args.sourceafis_template_cache_dir, repo_root=repo_root),
            request_timeout_seconds=args.request_timeout_seconds,
            extract_timeout_seconds=args.extract_timeout_seconds,
            verify_timeout_seconds=args.verify_timeout_seconds,
            max_retries=int(args.max_retries),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
            dpi_strategy=str(args.dpi_strategy),
            image_dpi=args.image_dpi,
            skip_sourceafis=bool(args.skip_sourceafis),
        )
    except (TrainScoreGenerationError, sourceafis.SourceAfisBenchmarkError) as exc:
        print(f"Train score generation failed: {exc}", file=sys.stderr)
        return 2

    print("Wrote train score artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
