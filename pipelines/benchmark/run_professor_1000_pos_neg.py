"""Superseded historical diagnostic benchmark.

This script is retained for reproducing the earlier Professor 1000 pos/neg
diagnostic run. Final comparable plain-vs-roll evidence should be produced
with `pipelines/benchmark/run_plain_roll_final_benchmark.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.benchmark import run_benchmark_matrix as matrix


DEFAULT_DATA_DIR = "data/manifests/nist_sd300b"
DEFAULT_OUTDIR = "artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg"
DEFAULT_THRESHOLD_SOURCE_DIR = (
    "artifacts/reports/benchmark/nist_sd300b_professor_1to1_five_methods_far_frr"
)
DEFAULT_SEED = 20260518
DEFAULT_N_PER_LABEL = 1000
TARGET_FAR = 0.01
METHODS = ["classic_v2", "minutiae", "harris", "sift", "dl_quick"]
DEPRECATION_NOTICE = (
    "SUPERSEDED LEGACY DIAGNOSTIC: kept only for reproducing the earlier "
    "Professor 1000 positive / 1000 negative artifact family. Use "
    "pipelines/benchmark/run_plain_roll_final_benchmark.py for current "
    "advisor-requested plain-vs-roll evidence."
)
PAIR_SETS = {
    "positive_1000": {"filename": "positive_1000.csv", "label": 1},
    "negative_1000": {"filename": "negative_1000.csv", "label": 0},
}
PAIR_COLUMNS = ["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"]


def resolve_path(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:"):]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _counts_for_pairs(df: pd.DataFrame) -> dict[str, Any]:
    labels = df["label"].astype(int)
    split_counts = df["split"].astype(str).value_counts().sort_index().to_dict() if "split" in df.columns else {}
    return {
        "n_rows": int(len(df)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "splits": {str(k): int(v) for k, v in split_counts.items()},
    }


def load_val_test_pairs(data_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source_paths = [data_dir / "pairs_val.csv", data_dir / "pairs_test.csv"]
    frames: list[pd.DataFrame] = []
    source_meta: list[dict[str, Any]] = []

    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required val/test pair source is missing: {path}")
        df = pd.read_csv(path)
        missing = [col for col in PAIR_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required pair columns: {missing}")
        splits = set(df["split"].astype(str).unique().tolist())
        if not splits.issubset({"val", "test"}):
            raise ValueError(f"{path} must contain only val/test rows; found splits={sorted(splits)}")
        df = df.copy()
        df["_source_pairs_file"] = str(path)
        frames.append(df)
        source_meta.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                **_counts_for_pairs(df),
            }
        )

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined, source_meta


def deterministic_sample_by_label(df: pd.DataFrame, *, label: int, n: int, seed: int) -> pd.DataFrame:
    labels = df["label"].astype(int)
    eligible = df[labels == int(label)].copy()
    if len(eligible) < int(n):
        raise ValueError(
            f"Not enough label={label} rows in val+test: need {int(n)}, found {len(eligible)}"
        )
    selected = eligible.sample(n=int(n), random_state=int(seed)).reset_index(drop=True)
    drop_cols = [col for col in selected.columns if col.startswith("_")]
    return selected.drop(columns=drop_cols)[PAIR_COLUMNS].copy()


def selected_pair_paths(outdir: Path) -> dict[str, Path]:
    selected_dir = outdir / "selected_pairs"
    return {
        pair_set: selected_dir / str(spec["filename"])
        for pair_set, spec in PAIR_SETS.items()
    }


def write_selected_pair_files(
    *,
    data_dir: Path,
    outdir: Path,
    seed: int = DEFAULT_SEED,
    n_per_label: int = DEFAULT_N_PER_LABEL,
) -> dict[str, Any]:
    combined, source_meta = load_val_test_pairs(data_dir)
    selected_dir = outdir / "selected_pairs"
    selected_dir.mkdir(parents=True, exist_ok=True)

    selected_meta: dict[str, Any] = {}
    for pair_set, spec in PAIR_SETS.items():
        selected = deterministic_sample_by_label(
            combined,
            label=int(spec["label"]),
            n=int(n_per_label),
            seed=int(seed),
        )
        bad_splits = set(selected["split"].astype(str).unique().tolist()) - {"val", "test"}
        if bad_splits:
            raise ValueError(f"{pair_set} includes non-val/test rows: {sorted(bad_splits)}")

        path = selected_dir / str(spec["filename"])
        selected.to_csv(path, index=False)
        selected_meta[pair_set] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "required_label": int(spec["label"]),
            **_counts_for_pairs(selected),
        }

    source_counts = _counts_for_pairs(combined)
    return {
        "seed": int(seed),
        "n_per_label": int(n_per_label),
        "source_files": source_meta,
        "source_counts": source_counts,
        "selected_pairs": selected_meta,
    }


def _threshold_for_far(negative_scores: pd.Series, target_far: float) -> tuple[float, int, float]:
    scores = pd.to_numeric(negative_scores, errors="coerce").dropna()
    if scores.empty:
        raise ValueError("Cannot calibrate threshold without finite negative scores.")

    n_negative = int(len(scores))
    for threshold in sorted(float(x) for x in scores.unique()):
        false_accepts = int((scores >= threshold).sum())
        far = false_accepts / n_negative
        if far <= float(target_far):
            return float(threshold), false_accepts, float(far)

    threshold = math.nextafter(float(scores.max()), math.inf)
    return threshold, 0, 0.0


def derive_thresholds_from_val_scores(
    *,
    source_dir: Path,
    methods: list[str] = METHODS,
    target_far: float = TARGET_FAR,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        score_path = source_dir / f"scores_{method}_val.csv"
        if not score_path.exists():
            raise FileNotFoundError(
                "Missing mixed validation score file required for FAR calibration: "
                f"{score_path}"
            )

        df = pd.read_csv(score_path)
        missing = {"label", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"{score_path} missing required score columns: {sorted(missing)}")

        labels = df["label"].astype(int)
        negatives = pd.to_numeric(df.loc[labels == 0, "score"], errors="coerce").dropna()
        positives = pd.to_numeric(df.loc[labels == 1, "score"], errors="coerce").dropna()
        if negatives.empty or positives.empty:
            raise ValueError(f"{score_path} must contain finite positive and negative val scores.")

        threshold, false_accepts, far = _threshold_for_far(negatives, target_far)
        true_accepts = int((positives >= threshold).sum())
        tar = true_accepts / int(len(positives))
        rows.append(
            {
                "method": method,
                "threshold": float(threshold),
                "threshold_source": str(score_path),
                "threshold_source_sha256": sha256_file(score_path),
                "target_far": float(target_far),
                "calibration_negatives": int(len(negatives)),
                "calibration_false_accepts": int(false_accepts),
                "calibration_far": float(far),
                "calibration_positives": int(len(positives)),
                "calibration_true_accepts": int(true_accepts),
                "calibration_tar": float(tar),
            }
        )
    return rows


def write_threshold_files(outdir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration_dir = outdir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    csv_path = calibration_dir / "thresholds_far_1pct_from_val.csv"
    json_path = calibration_dir / "thresholds_far_1pct_from_val.json"

    fieldnames = [
        "method",
        "threshold",
        "threshold_source",
        "threshold_source_sha256",
        "target_far",
        "calibration_negatives",
        "calibration_false_accepts",
        "calibration_far",
        "calibration_positives",
        "calibration_true_accepts",
        "calibration_tar",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_payload = {
        "schema_version": "v1_far_threshold_calibration",
        "target_far": TARGET_FAR,
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "rows": rows,
    }


def base_manifest(
    *,
    data_dir: Path,
    outdir: Path,
    selection: Mapping[str, Any],
    calibration: Mapping[str, Any],
    threshold_source_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "v1_professor_nist_sd300b_1000_pos_neg",
        "timestamp_utc": utc_now(),
        "request": "Professor-facing NIST SD300B positive-only and negative-only five-method benchmark.",
        "repo_root": str(ROOT),
        "outdir": str(outdir),
        "dataset": {
            "name": "nist_sd300b",
            "resolved_data_dir": str(data_dir),
        },
        "sampling": {
            "region": "val+test only",
            "excluded_region": "train",
            "seed": int(selection["seed"]),
            "n_per_label": int(selection["n_per_label"]),
            "source_files": selection["source_files"],
            "source_counts": selection["source_counts"],
            "selected_pairs": selection["selected_pairs"],
        },
        "methods": METHODS,
        "pair_sets": list(PAIR_SETS.keys()),
        "threshold_calibration": {
            "target_far": TARGET_FAR,
            "source_dir": str(threshold_source_dir),
            **dict(calibration),
        },
        "argv": list(sys.argv),
        "git": matrix.build_manifest_payload(
            dataset="nist_sd300b",
            data_dir=data_dir,
            outdir=outdir,
            methods=METHODS,
            splits=list(PAIR_SETS.keys()),
            limit=0,
            ensure_pairs=False,
            emb_cache_dir="",
            cache_write=False,
            cache_strip_prefix="",
            dedicated_ckpt="auto",
            fusion_fit_split="val",
            fusion_sift_weight=0.91,
            fusion_dl_weight=0.05,
            fusion_vit_weight=0.04,
            input_hashes={},
            mode="professor_pos_neg",
        ).get("git"),
    }


def write_manifest(outdir: Path, payload: Mapping[str, Any]) -> None:
    (outdir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_artifact_folder(
    *,
    data_dir: Path,
    outdir: Path,
    threshold_source_dir: Path,
    seed: int,
    n_per_label: int,
) -> dict[str, Any]:
    if outdir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark artifact folder: {outdir}"
        )
    outdir.mkdir(parents=True)

    selection = write_selected_pair_files(
        data_dir=data_dir,
        outdir=outdir,
        seed=seed,
        n_per_label=n_per_label,
    )
    threshold_rows = derive_thresholds_from_val_scores(
        source_dir=threshold_source_dir,
        methods=METHODS,
        target_far=TARGET_FAR,
    )
    calibration = write_threshold_files(outdir, threshold_rows)
    manifest = base_manifest(
        data_dir=data_dir,
        outdir=outdir,
        selection=selection,
        calibration=calibration,
        threshold_source_dir=threshold_source_dir,
    )
    write_manifest(outdir, manifest)
    return manifest


def load_threshold_rows(outdir: Path) -> dict[str, dict[str, Any]]:
    path = outdir / "calibration" / "thresholds_far_1pct_from_val.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration threshold CSV: {path}")
    df = pd.read_csv(path)
    return {str(row["method"]): row.to_dict() for _, row in df.iterrows()}


def _fail_if_output_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def run_all_methods(
    *,
    data_dir: Path,
    outdir: Path,
    emb_cache_dir: str,
    cache_write: bool,
    cache_strip_prefix: str,
) -> None:
    if not outdir.exists():
        raise FileNotFoundError(f"Artifact folder does not exist. Run --mode prepare first: {outdir}")
    paths = selected_pair_paths(outdir)
    for pair_set, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing selected pair file for {pair_set}: {path}")

    thresholds = load_threshold_rows(outdir)
    missing_methods = sorted(set(METHODS) - set(thresholds))
    if missing_methods:
        raise ValueError(f"Missing calibrated thresholds for methods: {missing_methods}")

    log_path = outdir / "run.log"
    _fail_if_output_exists(log_path)

    summary_csv = outdir / "results_summary.csv"
    summary_md = outdir / "results_summary.md"
    _fail_if_output_exists(summary_csv)
    _fail_if_output_exists(summary_md)

    def log(message: str) -> None:
        print(message)
        matrix.write_log_line(log_path, message)

    log("=== Professor NIST SD300B 1000 positive / 1000 negative benchmark ===")
    log(f"Repo root : {ROOT}")
    log(f"Data dir  : {data_dir}")
    log(f"Outdir    : {outdir}")
    log(f"Methods   : {METHODS}")
    log(f"Pair sets : {list(paths)}")

    commands: list[list[str]] = []
    for pair_set, pair_path in paths.items():
        for method in METHODS:
            threshold_row = thresholds[method]
            output_stem = f"{method}_{pair_set}"
            for path in [
                outdir / f"scores_{output_stem}.csv",
                outdir / f"run_{output_stem}.meta.json",
                outdir / f"roc_{output_stem}.png",
            ]:
                _fail_if_output_exists(path)

            threshold_source = (
                f"{threshold_row['threshold_source']} | target FAR <= {TARGET_FAR:.2%}"
            )
            cmd = matrix.build_eval_cmd(
                outdir=outdir,
                dataset="nist_sd300b",
                data_dir=data_dir,
                method=method,
                split="val",
                limit=0,
                ensure_pairs=False,
                dedicated_ckpt="auto",
                emb_cache_dir=emb_cache_dir,
                cache_write=cache_write,
                cache_strip_prefix=cache_strip_prefix,
                pairs_file=pair_path,
                pair_set_name=pair_set,
                operating_threshold=threshold_row["threshold"],
                threshold_source=threshold_source,
            )
            commands.append(cmd)
            log("[RUN] " + " ".join(cmd))
            rc = matrix.run_cmd_stream(cmd, cwd=ROOT, log_path=log_path)
            log(f"[DONE] method={method} pair_set={pair_set} exit_code={rc}")
            if rc != 0:
                raise RuntimeError(f"Benchmark command failed for {method}/{pair_set} with exit code {rc}")

    matrix.render_results_md(summary_csv, summary_md)
    log(f"OK: Wrote {summary_md}")

    manifest_path = outdir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"] = {
        "timestamp_utc": utc_now(),
        "commands": commands,
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": sha256_file(summary_csv),
        "summary_md": str(summary_md),
        "summary_md_sha256": sha256_file(summary_md),
        "run_log": str(log_path),
        "run_log_sha256": sha256_file(log_path),
    }
    write_manifest(outdir, manifest)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if math.isnan(value):  # type: ignore[arg-type]
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def verify_outputs(outdir: Path) -> None:
    problems: list[str] = []
    paths = selected_pair_paths(outdir)
    for pair_set, path in paths.items():
        if not path.exists():
            problems.append(f"missing selected pair file: {path}")
            continue
        df = pd.read_csv(path)
        expected_label = int(PAIR_SETS[pair_set]["label"])
        if len(df) != DEFAULT_N_PER_LABEL:
            problems.append(f"{pair_set}: expected {DEFAULT_N_PER_LABEL} rows, found {len(df)}")
        if set(df["label"].astype(int).unique().tolist()) != {expected_label}:
            problems.append(f"{pair_set}: expected only label={expected_label}")
        splits = set(df["split"].astype(str).unique().tolist())
        if not splits.issubset({"val", "test"}):
            problems.append(f"{pair_set}: selected rows must be val/test only; found {sorted(splits)}")
        if "train" in splits:
            problems.append(f"{pair_set}: train rows are not allowed")

    threshold_csv = outdir / "calibration" / "thresholds_far_1pct_from_val.csv"
    threshold_json = outdir / "calibration" / "thresholds_far_1pct_from_val.json"
    if not threshold_csv.exists():
        problems.append(f"missing threshold CSV: {threshold_csv}")
    if not threshold_json.exists():
        problems.append(f"missing threshold JSON: {threshold_json}")
    if threshold_csv.exists():
        threshold_df = pd.read_csv(threshold_csv)
        if set(threshold_df["method"].astype(str).tolist()) != set(METHODS):
            problems.append("threshold CSV does not contain exactly the five official methods")

    summary_csv = outdir / "results_summary.csv"
    summary_md = outdir / "results_summary.md"
    run_manifest = outdir / "run_manifest.json"
    run_log = outdir / "run.log"
    for path in [summary_csv, summary_md, run_manifest, run_log]:
        if not path.exists():
            problems.append(f"missing artifact: {path}")

    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        expected_combos = {(method, pair_set) for method in METHODS for pair_set in PAIR_SETS}
        actual_combos = set(zip(summary["method"].astype(str), summary["split"].astype(str)))
        if actual_combos != expected_combos:
            problems.append(f"summary method/pair-set combos mismatch: {sorted(actual_combos)}")
        if len(summary) != len(expected_combos):
            problems.append(f"summary should contain {len(expected_combos)} rows, found {len(summary)}")

        for _, row in summary.iterrows():
            method = str(row["method"])
            pair_set = str(row["split"])
            scores_path = outdir / f"scores_{method}_{pair_set}.csv"
            meta_path = outdir / f"run_{method}_{pair_set}.meta.json"
            roc_path = outdir / f"roc_{method}_{pair_set}.png"

            if not scores_path.exists():
                problems.append(f"missing scores CSV: {scores_path}")
            else:
                scores = pd.read_csv(scores_path)
                if len(scores) != DEFAULT_N_PER_LABEL:
                    problems.append(f"{scores_path}: expected {DEFAULT_N_PER_LABEL} rows, found {len(scores)}")
                expected_label = int(PAIR_SETS[pair_set]["label"])
                if set(scores["label"].astype(int).unique().tolist()) != {expected_label}:
                    problems.append(f"{scores_path}: expected only label={expected_label}")

            if not meta_path.exists():
                problems.append(f"missing run meta: {meta_path}")
            else:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                roc = meta.get("roc", {})
                if roc.get("status") != "skipped":
                    problems.append(f"{meta_path}: ROC status should be skipped for single-label input")
                if "only one label" not in str(roc.get("skip_reason", "")):
                    problems.append(f"{meta_path}: missing one-label ROC skip reason")

            if roc_path.exists():
                problems.append(f"ROC PNG should be skipped for single-label run, but exists: {roc_path}")

            if int(row["n_pairs"]) != DEFAULT_N_PER_LABEL:
                problems.append(f"{method}/{pair_set}: n_pairs should be {DEFAULT_N_PER_LABEL}")
            if _is_missing(row.get("threshold")):
                problems.append(f"{method}/{pair_set}: threshold is missing")
            if _is_missing(row.get("total_runtime_seconds")):
                problems.append(f"{method}/{pair_set}: total runtime is missing")

            if pair_set == "positive_1000":
                if int(row.get("n_positive", -1)) != DEFAULT_N_PER_LABEL or int(row.get("n_negative", -1)) != 0:
                    problems.append(f"{method}/{pair_set}: label counts are incorrect")
                if int(row.get("accepted_count", -1)) + int(row.get("rejected_count", -1)) != DEFAULT_N_PER_LABEL:
                    problems.append(f"{method}/{pair_set}: accepted+rejected count mismatch")
                if _is_missing(row.get("tar")) or _is_missing(row.get("frr")):
                    problems.append(f"{method}/{pair_set}: TAR/FRR should be populated")
                if not _is_missing(row.get("far")):
                    problems.append(f"{method}/{pair_set}: FAR should be N/A")
            elif pair_set == "negative_1000":
                if int(row.get("n_positive", -1)) != 0 or int(row.get("n_negative", -1)) != DEFAULT_N_PER_LABEL:
                    problems.append(f"{method}/{pair_set}: label counts are incorrect")
                if int(row.get("false_accept_count", -1)) + int(row.get("true_reject_count", -1)) != DEFAULT_N_PER_LABEL:
                    problems.append(f"{method}/{pair_set}: false_accept+true_reject count mismatch")
                if _is_missing(row.get("far")):
                    problems.append(f"{method}/{pair_set}: FAR should be populated")
                if not _is_missing(row.get("tar")) or not _is_missing(row.get("frr")):
                    problems.append(f"{method}/{pair_set}: TAR/FRR should be N/A")

    if problems:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(problems))

    ok_path = outdir / "validation.ok"
    ok_path.write_text(
        "OK\n"
        f"timestamp_utc: {utc_now()}\n"
        f"outdir       : {outdir}\n",
        encoding="utf-8",
    )
    print(f"[OK] Professor benchmark artifacts verified: {outdir}")
    print(f"[OK] Wrote {ok_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY historical diagnostic: run Professor Menachem's NIST SD300B 1000 pos/neg benchmark. "
            "For final comparable plain-vs-roll evidence, use "
            "pipelines/benchmark/run_plain_roll_final_benchmark.py."
        )
    )
    parser.add_argument("--mode", choices=["all", "prepare", "run", "verify"], default="all")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--threshold_source_dir", default=DEFAULT_THRESHOLD_SOURCE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_per_label", type=int, default=DEFAULT_N_PER_LABEL)
    parser.add_argument(
        "--emb_cache_dir",
        default="",
        help="Optional persistent DL embedding cache. Empty by default so timing is not affected by prior runs.",
    )
    parser.add_argument("--cache_write", action="store_true")
    parser.add_argument("--cache_strip_prefix", default="")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    print(f"WARNING: {DEPRECATION_NOTICE}", file=sys.stderr)
    args = build_parser().parse_args(argv)
    data_dir = resolve_path(args.data_dir)
    outdir = resolve_path(args.outdir)
    threshold_source_dir = resolve_path(args.threshold_source_dir)
    emb_cache_dir = str(resolve_path(args.emb_cache_dir)) if args.emb_cache_dir else ""

    if args.mode in {"all", "prepare"}:
        prepare_artifact_folder(
            data_dir=data_dir,
            outdir=outdir,
            threshold_source_dir=threshold_source_dir,
            seed=int(args.seed),
            n_per_label=int(args.n_per_label),
        )

    if args.mode in {"all", "run"}:
        run_all_methods(
            data_dir=data_dir,
            outdir=outdir,
            emb_cache_dir=emb_cache_dir,
            cache_write=bool(args.cache_write),
            cache_strip_prefix=str(args.cache_strip_prefix),
        )

    if args.mode in {"all", "verify"}:
        verify_outputs(outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
