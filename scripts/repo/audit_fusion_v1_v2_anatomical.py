from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_fusion_v1_v2_anatomical_full_pairs"
)
DATASETS = ("nist_sd300b", "nist_sd300c")
SPLITS = ("val", "test")
METHOD = "sourceafis_sift_quality_fusion_v1"
EXPECTED = {
    "train": {"rows": 28052, "pos": 7013, "neg": 21039},
    "val": {"rows": 3508, "pos": 877, "neg": 2631},
    "test": {"rows": 3556, "pos": 889, "neg": 2667},
}


class AuditFailure(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _score_meta_candidates(score_csv: Path) -> list[Path]:
    return [score_csv.with_suffix(".meta.json"), Path(str(score_csv) + ".meta.json")]


def _score_meta_path(score_csv: Path) -> Path:
    for candidate in _score_meta_candidates(score_csv):
        if candidate.exists():
            return candidate
    raise AuditFailure(f"missing score meta JSON for {score_csv}")


def _run_meta_path(outdir: Path, dataset: str, split: str) -> Path:
    return outdir / "run_meta" / f"run_{dataset}_{METHOD}_{split}.meta.json"


def _labels(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)


def _frgp(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["frgp"], errors="coerce").fillna(-1).astype(int)


def _capture_code(path: Any) -> tuple[str, str]:
    parts = Path(str(path)).name.lower().split("_")
    if len(parts) < 4:
        return "", ""
    capture = parts[-3]
    code = parts[-1].split(".")[0]
    if capture == "rolled":
        capture = "roll"
    if capture not in {"plain", "roll"}:
        return "", ""
    return capture, code


def _plain_roll_codes(row: pd.Series) -> tuple[str, str]:
    cap_a, code_a = _capture_code(row["path_a"])
    cap_b, code_b = _capture_code(row["path_b"])
    plain = code_a if cap_a == "plain" else code_b if cap_b == "plain" else ""
    roll = code_a if cap_a == "roll" else code_b if cap_b == "roll" else ""
    return plain, roll


def _normalize_for_alignment(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "finger_position" in out.columns and "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]
    required = {"dataset", "split", "pair_id", "label", "subject_a", "subject_b", "path_a", "path_b", "frgp"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise AuditFailure(f"{dataset}/{split}: table missing required columns {missing}")
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    out["frgp"] = pd.to_numeric(out["frgp"], errors="coerce").fillna(-1).astype(int)
    for column in ("subject_a", "subject_b", "path_a", "path_b"):
        out[column] = out[column].astype(str).str.strip()
    return out[(out["dataset"] == dataset) & (out["split"] == split)].copy()


def _alignment_audit(*, dataset: str, split: str, pairs: pd.DataFrame, scores: pd.DataFrame) -> None:
    key = ["dataset", "split", "pair_id"]
    pair_keys = pairs[key].copy()
    score_keys = scores[key].copy()
    merged = pair_keys.merge(score_keys, on=key, how="outer", indicator=True, validate="one_to_one")
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        raise AuditFailure(f"{dataset}/{split}: score pair key mismatch missing={missing} extra={extra}")

    compare = pairs[key + ["label", "subject_a", "subject_b", "path_a", "path_b", "frgp"]].merge(
        scores[key + ["label", "subject_a", "subject_b", "path_a", "path_b", "frgp"]],
        on=key,
        suffixes=("_pair", "_score"),
        validate="one_to_one",
    )
    bad: dict[str, int] = {}
    for column in ("label", "subject_a", "subject_b", "path_a", "path_b", "frgp"):
        left = compare[f"{column}_pair"].astype(str)
        right = compare[f"{column}_score"].astype(str)
        mismatches = int((left != right).sum())
        if mismatches:
            bad[column] = mismatches
    if bad:
        raise AuditFailure(f"{dataset}/{split}: selected_pairs/scores alignment mismatches {bad}")


def _pair_stats(df: pd.DataFrame) -> dict[str, Any]:
    labels = _labels(df)
    frgp = _frgp(df)
    return {
        "rows": int(len(df)),
        "pos": int((labels == 1).sum()),
        "neg": int((labels == 0).sum()),
        "frgp_coverage": sorted(set(int(value) for value in frgp[frgp > 0].tolist())),
        "frgp_1_pos": int(((frgp == 1) & (labels == 1)).sum()),
        "frgp_1_neg": int(((frgp == 1) & (labels == 0)).sum()),
        "frgp_6_pos": int(((frgp == 6) & (labels == 1)).sum()),
        "frgp_6_neg": int(((frgp == 6) & (labels == 0)).sum()),
    }


def _anatomical_pair_checks(df: pd.DataFrame) -> dict[str, Any]:
    labels = _labels(df)
    codes = [_plain_roll_codes(row) for _, row in df.iterrows()]
    positives = [code for code, label in zip(codes, labels.tolist()) if int(label) == 1]
    plain_codes = [plain for plain, _roll in codes if plain]
    return {
        "plain_13_plain_14_refs": int(sum(1 for code in plain_codes if code in {"13", "14"})),
        "plain_11_roll_01_positive_count": int(sum(1 for item in positives if item == ("11", "01"))),
        "plain_12_roll_06_positive_count": int(sum(1 for item in positives if item == ("12", "06"))),
        "plain_11_roll_11_count": int(sum(1 for item in codes if item == ("11", "11"))),
        "plain_12_roll_12_count": int(sum(1 for item in codes if item == ("12", "12"))),
    }


def _check_training_manifest(outdir: Path) -> dict[str, Any]:
    manifest_path = outdir / "model" / "training_manifest.json"
    manifest = _read_json(manifest_path)
    required_false = ("test_used_for_training", "legacy_scores_used", "artifact_selected_pairs_used_as_input")
    for field in required_false:
        if manifest.get(field) is not False:
            raise AuditFailure(f"training_manifest field {field} is not false")
    if manifest.get("trained_on_splits") != ["train"]:
        raise AuditFailure("training_manifest trained_on_splits is not ['train']")
    if manifest.get("thresholds_selected_on") != "val":
        raise AuditFailure("training_manifest thresholds_selected_on is not val")
    if manifest.get("run_pair_bundle_version") != "sd300_anatomical_full_pairs_v2":
        raise AuditFailure("training_manifest run_pair_bundle_version mismatch")
    if manifest.get("sd300_frgp_semantics") != "anatomical":
        raise AuditFailure("training_manifest sd300_frgp_semantics mismatch")
    if int(manifest.get("training_rows", -1)) != EXPECTED["train"]["rows"] * len(DATASETS):
        raise AuditFailure("training_manifest training_rows mismatch")
    dataset_counts = {str(k): int(v) for k, v in dict(manifest.get("dataset_counts", {})).items()}
    for dataset in DATASETS:
        if dataset_counts.get(dataset) != EXPECTED["train"]["rows"]:
            raise AuditFailure(f"training_manifest dataset_counts mismatch for {dataset}: {dataset_counts.get(dataset)}")
    return {
        "path": str(manifest_path),
        "training_rows": int(manifest.get("training_rows", 0)),
        "dataset_counts": dataset_counts,
        "feature_schema_sha256": manifest.get("feature_schema_sha256", ""),
        "model_file_sha256": manifest.get("model_file_sha256", ""),
    }


def _check_metrics(outdir: Path) -> dict[str, Any]:
    metrics_path = outdir / "plain_roll_final_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    expected_rows = len(DATASETS) * len(SPLITS) * 2
    if len(metrics) != expected_rows:
        raise AuditFailure(f"metrics row count mismatch expected={expected_rows} actual={len(metrics)}")
    for column in ("auc", "eer", "tar", "far", "threshold"):
        values = pd.to_numeric(metrics[column], errors="coerce")
        if not values.map(math.isfinite).all():
            raise AuditFailure(f"metrics column {column} contains non-finite values")

    comparison_path = outdir / "plain_roll_final_sourceafis_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    if len(comparison) != expected_rows:
        raise AuditFailure(f"comparison row count mismatch expected={expected_rows} actual={len(comparison)}")
    return {
        "metrics_csv": str(metrics_path),
        "comparison_csv": str(comparison_path),
        "metrics_rows": int(len(metrics)),
        "comparison_rows": int(len(comparison)),
    }


def audit_output(outdir: Path) -> dict[str, Any]:
    if not outdir.exists():
        raise AuditFailure(f"output directory does not exist: {outdir}")
    name = outdir.name
    for token in ("fusion_v1", "v2", "anatomical", "full_pairs"):
        if token not in name:
            raise AuditFailure(f"output directory name is missing token {token!r}: {name}")

    training = _check_training_manifest(outdir)
    score_reports: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for split in SPLITS:
            selected_csv = outdir / "selected_pairs" / f"pairs_{dataset}_{split}.csv"
            score_csv = outdir / "scores" / f"scores_{dataset}_{METHOD}_{split}.csv"
            pairs = _normalize_for_alignment(pd.read_csv(selected_csv), dataset=dataset, split=split)
            scores = _normalize_for_alignment(pd.read_csv(score_csv), dataset=dataset, split=split)
            _alignment_audit(dataset=dataset, split=split, pairs=pairs, scores=scores)
            stats = _pair_stats(scores)
            expected = EXPECTED[split]
            if stats["rows"] != expected["rows"] or stats["pos"] != expected["pos"] or stats["neg"] != expected["neg"]:
                raise AuditFailure(f"{dataset}/{split}: score counts mismatch {stats} expected={expected}")
            if stats["frgp_coverage"] != list(range(1, 11)):
                raise AuditFailure(f"{dataset}/{split}: frgp coverage mismatch {stats['frgp_coverage']}")
            if min(stats["frgp_1_pos"], stats["frgp_1_neg"], stats["frgp_6_pos"], stats["frgp_6_neg"]) <= 0:
                raise AuditFailure(f"{dataset}/{split}: frgp 1/6 positive-negative coverage missing")

            anatomical = _anatomical_pair_checks(scores)
            if anatomical["plain_13_plain_14_refs"] != 0:
                raise AuditFailure(f"{dataset}/{split}: plain_13/plain_14 refs found")
            if anatomical["plain_11_roll_01_positive_count"] <= 0:
                raise AuditFailure(f"{dataset}/{split}: missing plain_11->roll_01 positives")
            if anatomical["plain_12_roll_06_positive_count"] <= 0:
                raise AuditFailure(f"{dataset}/{split}: missing plain_12->roll_06 positives")
            if anatomical["plain_11_roll_11_count"] or anatomical["plain_12_roll_12_count"]:
                raise AuditFailure(f"{dataset}/{split}: legacy plain-to-same-roll thumb references found")

            meta = _read_json(_score_meta_path(score_csv))
            run_meta = _read_json(_run_meta_path(outdir, dataset, split))
            for payload, label in ((meta, "score_meta"), (run_meta, "run_meta")):
                if payload.get("pair_source_sha256") != scores["pair_source_sha256"].iloc[0]:
                    raise AuditFailure(f"{dataset}/{split}: {label} pair_source_sha256 mismatch")
                if payload.get("sd300_frgp_semantics") != "anatomical":
                    raise AuditFailure(f"{dataset}/{split}: {label} sd300_frgp_semantics mismatch")
                if payload.get("run_pair_bundle_version") != "sd300_anatomical_full_pairs_v2":
                    raise AuditFailure(f"{dataset}/{split}: {label} run_pair_bundle_version mismatch")

            score_reports.append({"dataset": dataset, "split": split, **stats, **anatomical})

    manifest = _read_json(outdir / "plain_roll_final_manifest.json")
    if manifest.get("legacy_scores_used") is not False:
        raise AuditFailure("final manifest legacy_scores_used is not false")
    if manifest.get("artifact_selected_pairs_used_as_input") is not False:
        raise AuditFailure("final manifest artifact_selected_pairs_used_as_input is not false")
    if manifest.get("test_used_for_training") is not False:
        raise AuditFailure("final manifest test_used_for_training is not false")
    failed_sources = [
        row for row in manifest.get("source_score_alignment_checks", []) if row.get("status") != "pass"
    ]
    if failed_sources:
        raise AuditFailure(f"source score alignment checks did not all pass: {failed_sources[:3]}")

    metrics = _check_metrics(outdir)
    return {
        "schema_version": "fusion_v1_v2_anatomical_output_audit_v1",
        "created_at": _utc_now(),
        "outdir": str(outdir),
        "training": training,
        "scores": score_reports,
        "metrics": metrics,
        "status": "pass",
    }


def _write_reports(outdir: Path, payload: dict[str, Any]) -> None:
    audit_dir = outdir / "pair_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "fusion_v1_v2_anatomical_output_audit.json"
    md_path = audit_dir / "fusion_v1_v2_anatomical_output_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Fusion v1 v2 Anatomical Output Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- output: `{payload['outdir']}`",
        f"- training rows: `{payload['training']['training_rows']}`",
        f"- metrics rows: `{payload['metrics']['metrics_rows']}`",
        f"- comparison rows: `{payload['metrics']['comparison_rows']}`",
        "",
        "| dataset | split | rows | pos | neg | frgp | plain11-roll01 pos | plain12-roll06 pos |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in payload["scores"]:
        lines.append(
            "| {dataset} | {split} | {rows} | {pos} | {neg} | {frgp} | {p1101} | {p1206} |".format(
                dataset=row["dataset"],
                split=row["split"],
                rows=row["rows"],
                pos=row["pos"],
                neg=row["neg"],
                frgp=",".join(str(item) for item in row["frgp_coverage"]),
                p1101=row["plain_11_roll_01_positive_count"],
                p1206=row["plain_12_roll_06_positive_count"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Fusion v1 v2 SD300 anatomical full-pairs output.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args(argv)
    outdir = Path(args.outdir).resolve()
    try:
        payload = audit_output(outdir)
    except Exception as exc:
        print(f"Fusion v1 v2 anatomical audit failed: {exc}", file=sys.stderr)
        return 2
    _write_reports(outdir, payload)
    print("Fusion v1 v2 anatomical audit passed.")
    print(f"output: {outdir}")
    print(f"metrics: {payload['metrics']['metrics_csv']}")
    print(f"comparison: {payload['metrics']['comparison_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
