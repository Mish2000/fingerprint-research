"""Frozen SD300 deep pair reranker -> zero-shot scoring on PolyU Cross (Phase 3A.2).

Scores the *existing* PolyU Cross pair bundle with the *existing* frozen SD300
``deep_pair_reranker_fast_ddp`` checkpoint. It reuses the canonical model
architecture, image preprocessing, and scoring loop from
``scripts/deep/score_fast_pair_ddp_splits.py`` (the code that produced the
SD300 anatomical-v2 deep scores) without rewriting either the model or the
preprocessing. The only new work is (a) resolving PolyU relative image paths
against the configured dataset root using the Phase 3A loader and (b)
enriching/auditing the produced scores.

Frozen guarantees
-----------------
* The checkpoint is loaded with ``strict=True`` into the canonical ``PairModel``.
* ``model.eval()`` and every parameter has ``requires_grad_(False)``.
* Inference runs under ``torch.no_grad`` (via ``score_table``).
* No optimizer, scheduler, loss, or fitting of any kind is instantiated.

It deliberately does **not** train, fine-tune, calibrate, choose thresholds,
run fusion, regenerate pairs, edit the manifest/pairs, or touch UI/API code.

Outputs (under ``--outdir``)
----------------------------
* ``scores_polyu_cross_deep_pair_reranker_<split>.csv``   - enriched scores.
* ``failures_polyu_cross_deep_pair_reranker_<split>.csv`` - unreadable pairs.
* ``run_polyu_cross_deep_pair_reranker_<split>.meta.json``- per-split run meta.
* ``latency_summary.csv``                                 - per-split latency.
* ``run_manifest.json``                                   - whole-run manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    PolyUCrossPairError,
    balanced_limit,
    iter_pair_split_csvs,
    load_polyu_cross_pairs,
    resolve_polyu_cross_root,
    resolve_pairs_frame,
)

# Reuse the small provenance helpers from the Phase 3A runner for consistency.
from pipelines.benchmark.run_polyu_cross_zero_shot import (
    ensure_dir,
    git_info,
    resolve_repo_path,
    safe_pkg_version,
    sha256_file,
    utc_now,
)

DATASET_NAME = POLYU_CROSS_DATASET
# Canonical SD300 anatomical-v2 checkpoint (fast_ddp pair model, NIST-only).
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_OUTDIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_deep_v0"

# File token is fixed by the task spec; the ``method`` column keeps the canonical id.
METHOD_FILE_TOKEN = "deep_pair_reranker"
METHOD = "deep_pair_reranker_fast_ddp"
RUN_SCHEMA_VERSION = "polyu_cross_zero_shot_deep_v0"

# Preprocessing identifier that matches score_fast_pair_ddp_splits.load_image_u8 +
# score_table: grayscale ("L") -> resize to input_size x input_size (stretch,
# bilinear) -> uint8 -> float/255 in [0, 1]. No foreground crop, no [-1, 1]
# normalization, no inversion. This is the checkpoint's existing input contract.
PREPROCESS_ID = "fast_ddp_grayscale_resize_stretch_div255"

REQUIRED_SCORE_COLUMNS = [
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "sample_uid_a",
    "sample_uid_b",
    "modality_a",
    "modality_b",
    "session_a",
    "session_b",
    "score",
    "method",
    "dataset",
]

EXTRA_IDENTITY_COLUMNS = ["finger_unit_a", "finger_unit_b", "frgp", "path_a", "path_b"]
RESOLUTION_COLUMNS = ["resolved_path_a", "resolved_path_b", "path_a_exists", "path_b_exists", "status", "error"]
DIAG_COLUMNS = ["logit", "probability", "score_semantics", "higher_is_more_similar", "pair_total_ms"]

FAILURE_COLUMNS = [
    "method",
    "dataset",
    "split",
    "pair_id",
    "label",
    "subject_a",
    "subject_b",
    "path_a",
    "path_b",
    "resolved_path_a",
    "resolved_path_b",
    "operation",
    "error_type",
    "error_message",
]

LATENCY_COLUMNS = [
    "method",
    "split",
    "n_pairs",
    "n_scored",
    "n_failed",
    "avg_ms_pair",
    "total_ms",
    "n_unique_images",
    "device",
    "source",
]


class PolyUCrossDeepRunError(RuntimeError):
    """Raised for unrecoverable deep-runner setup/protocol failures."""


# ---------------------------------------------------------------------------
# Frozen model + scoring (reuses the canonical fast_ddp inference code)
# ---------------------------------------------------------------------------
@dataclass
class FrozenModel:
    model: Any
    device: Any
    input_size: int
    checkpoint: Path
    provenance: dict[str, Any]


def load_frozen_model(checkpoint: Path, device_arg: str = "auto") -> FrozenModel:
    """Load the canonical fast_ddp ``PairModel`` frozen for inference.

    Reuses ``PairModel`` / ``safe_torch_load`` from the existing SD300 deep score
    generator. Enforces strict state-dict loading, eval mode, and disabled
    gradients. Raises if the checkpoint/model are incompatible.
    """

    import torch

    from scripts.deep.score_fast_pair_ddp_splits import PairModel, safe_torch_load

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise PolyUCrossDeepRunError(f"Checkpoint not found: {checkpoint}")

    payload = safe_torch_load(checkpoint)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise PolyUCrossDeepRunError(
            f"Checkpoint {checkpoint} does not contain a 'model_state_dict'; refusing to guess architecture."
        )
    args = dict(payload.get("args", {}) or {})
    width = int(args.get("width", 32))
    embedding_dim = int(args.get("embedding_dim", 512))
    hidden_dim = int(args.get("hidden_dim", 768))
    input_size = int(args.get("input_size", 384))

    if str(device_arg).lower() in ("auto", ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        want_cuda = str(device_arg).lower().startswith("cuda")
        device = torch.device(device_arg if (want_cuda and torch.cuda.is_available()) else "cpu")

    model = PairModel(width=width, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    # strict=True: the SD300 checkpoint must map exactly onto the frozen model.
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": payload.get("model_type"),
        "model_identifier": f"fast_ddp_PairModel(width={width},embedding_dim={embedding_dim},hidden_dim={hidden_dim})",
        "architecture": {
            "family": "PairModel",
            "width": width,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "param_count": int(sum(p.numel() for p in model.parameters())),
        },
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics"),
        "checkpoint_args": args,
        "preprocess_identifier": PREPROCESS_ID,
        "preprocess_config": {
            "grayscale": True,
            "resize_to": [input_size, input_size],
            "resize_mode": "bilinear_stretch_no_pad",
            "foreground_crop": False,
            "value_range": "div255_0_1",
            "invert": False,
            "channels": 1,
        },
        "device": device.type,
        "frozen": True,
        "trained_on": {
            "model_type": payload.get("model_type"),
            "datasets": ["nist_sd300b", "nist_sd300c"],
            "note": "Trained on NIST SD300 plain/roll pairs (contact-based); PolyU Cross is unseen zero-shot transfer.",
        },
        "no_grad": True,
        "eval_mode": True,
    }
    return FrozenModel(model=model, device=device, input_size=input_size, checkpoint=checkpoint, provenance=provenance)


def score_resolved_pairs(
    frozen: FrozenModel,
    resolved_scorable: pd.DataFrame,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Score scorable pairs (both images present) with the frozen model.

    Reuses ``load_image_u8`` (exact checkpoint preprocessing) and ``score_table``
    (exact frozen forward + sigmoid) from the SD300 deep score generator.
    Returns ``(logits, probs, timing)`` aligned to ``resolved_scorable`` order.
    """

    import torch

    from scripts.deep.score_fast_pair_ddp_splits import load_image_u8, score_table

    n = len(resolved_scorable)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), {
            "total_ms": 0.0,
            "avg_ms_pair": float("nan"),
            "n_unique_images": 0,
        }

    input_size = int(frozen.input_size)
    paths_a = resolved_scorable["resolved_path_a"].astype(str).tolist()
    paths_b = resolved_scorable["resolved_path_b"].astype(str).tolist()
    unique_paths = sorted(set(paths_a).union(paths_b))
    path_to_idx = {p: i for i, p in enumerate(unique_paths)}

    images = torch.empty((len(unique_paths), 1, input_size, input_size), dtype=torch.uint8)
    for path in unique_paths:
        images[path_to_idx[path]] = load_image_u8(Path(path), input_size)

    pair_idx = torch.empty((n, 2), dtype=torch.long)
    for i in range(n):
        pair_idx[i, 0] = path_to_idx[paths_a[i]]
        pair_idx[i, 1] = path_to_idx[paths_b[i]]

    start = time.perf_counter()
    logits, probs = score_table(
        frozen.model,
        images,
        pair_idx,
        device=frozen.device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        amp=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timing = {
        "total_ms": float(elapsed_ms),
        "avg_ms_pair": float(elapsed_ms / max(n, 1)),
        "n_unique_images": int(len(unique_paths)),
    }
    return np.asarray(logits, dtype=float), np.asarray(probs, dtype=float), timing


# ---------------------------------------------------------------------------
# Enrichment / failures
# ---------------------------------------------------------------------------
def build_enriched_scores(
    resolved_df: pd.DataFrame,
    *,
    split: str,
    scores: np.ndarray,
    logits: np.ndarray,
    status: list[str],
    errors: list[str],
    per_pair_ms: np.ndarray,
) -> pd.DataFrame:
    n = len(resolved_df)
    frame: dict[str, Any] = {
        "pair_id": resolved_df["pair_id"].values,
        "label": resolved_df["label"].astype(int).values,
        "split": resolved_df["split"].astype(str).values if "split" in resolved_df else [split] * n,
        "subject_a": resolved_df["subject_a"].values,
        "subject_b": resolved_df["subject_b"].values,
        "score": np.asarray(scores, dtype=float),
        "method": [METHOD] * n,
        "dataset": [DATASET_NAME] * n,
    }
    for column in ("sample_uid_a", "sample_uid_b", "modality_a", "modality_b", "session_a", "session_b"):
        frame[column] = resolved_df[column].values if column in resolved_df.columns else [""] * n
    for column in EXTRA_IDENTITY_COLUMNS:
        if column in resolved_df.columns:
            frame[column] = resolved_df[column].values
    for column in ("resolved_path_a", "resolved_path_b", "path_a_exists", "path_b_exists"):
        if column in resolved_df.columns:
            frame[column] = resolved_df[column].values
    frame["status"] = status
    frame["error"] = errors
    frame["logit"] = np.asarray(logits, dtype=float)
    frame["probability"] = np.asarray(scores, dtype=float)
    frame["score_semantics"] = ["deep_pair_match_probability"] * n
    frame["higher_is_more_similar"] = [True] * n
    frame["pair_total_ms"] = np.asarray(per_pair_ms, dtype=float)

    df = pd.DataFrame(frame)
    leading = [c for c in REQUIRED_SCORE_COLUMNS if c in df.columns]
    identity = [c for c in EXTRA_IDENTITY_COLUMNS if c in df.columns]
    resolution = [c for c in RESOLUTION_COLUMNS if c in df.columns]
    diag = [c for c in DIAG_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in (leading + identity + resolution + diag)]
    return df[leading + identity + resolution + diag + rest]


def failures_from_scores(scored: pd.DataFrame, *, split: str) -> pd.DataFrame:
    failed = scored[scored["status"].astype(str) != "ok"]
    rows: list[dict[str, Any]] = []
    for _, row in failed.iterrows():
        rows.append(
            {
                "method": METHOD,
                "dataset": DATASET_NAME,
                "split": split,
                "pair_id": row.get("pair_id", ""),
                "label": row.get("label", ""),
                "subject_a": row.get("subject_a", ""),
                "subject_b": row.get("subject_b", ""),
                "path_a": row.get("path_a", ""),
                "path_b": row.get("path_b", ""),
                "resolved_path_a": row.get("resolved_path_a", ""),
                "resolved_path_b": row.get("resolved_path_b", ""),
                "operation": "score_pair",
                "error_type": "unscorable_pair",
                "error_message": str(row.get("error", "")),
            }
        )
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


# ---------------------------------------------------------------------------
# Per-split driver
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    split: str
    pairs_csv: Path
    scores_csv: Path
    run_meta_json: Path
    failures_csv: Path
    n_pairs: int
    n_scored: int
    n_failed: int
    n_positive: int
    n_negative: int
    timing: dict[str, Any]


def run_split(
    *,
    frozen: FrozenModel,
    split: str,
    pairs_csv: Path,
    root: Optional[Path],
    outdir: Path,
    limit: int,
    strict: bool,
    batch_size: int,
    num_workers: int,
) -> SplitResult:
    pairs = load_polyu_cross_pairs(pairs_csv)
    if int(limit) > 0:
        pairs = balanced_limit(pairs, int(limit))
    resolved = resolve_pairs_frame(pairs, root).reset_index(drop=True)

    exists_a = resolved.get("path_a_exists", pd.Series([True] * len(resolved))).astype(bool).to_numpy()
    exists_b = resolved.get("path_b_exists", pd.Series([True] * len(resolved))).astype(bool).to_numpy()
    scorable_mask = exists_a & exists_b

    if strict and not scorable_mask.all():
        missing = resolved[~scorable_mask]
        raise PolyUCrossDeepRunError(
            f"[strict] {len(missing)} pair(s) reference missing images for split={split}. "
            f"First: {missing.iloc[0]['resolved_path_a']!r} / {missing.iloc[0]['resolved_path_b']!r}"
        )

    n = len(resolved)
    scores = np.full(n, np.nan, dtype=float)
    logits = np.full(n, np.nan, dtype=float)
    per_pair_ms = np.full(n, np.nan, dtype=float)

    scorable = resolved[scorable_mask].reset_index(drop=True)
    timing = {"total_ms": 0.0, "avg_ms_pair": float("nan"), "n_unique_images": 0}
    if len(scorable) > 0:
        s_logits, s_probs, timing = score_resolved_pairs(
            frozen, scorable, batch_size=batch_size, num_workers=num_workers
        )
        scores[scorable_mask] = s_probs
        logits[scorable_mask] = s_logits
        per_pair_ms[scorable_mask] = timing.get("avg_ms_pair", float("nan"))

    status: list[str] = []
    errors: list[str] = []
    for i in range(n):
        if not scorable_mask[i]:
            missing = []
            if not exists_a[i]:
                missing.append("path_a")
            if not exists_b[i]:
                missing.append("path_b")
            status.append("failed")
            errors.append(f"missing_image:{'+'.join(missing)}")
        elif not np.isfinite(scores[i]):
            status.append("failed")
            errors.append("non_finite_score")
        else:
            status.append("ok")
            errors.append("")

    enriched = build_enriched_scores(
        resolved, split=split, scores=scores, logits=logits, status=status, errors=errors, per_pair_ms=per_pair_ms
    )

    scores_csv = outdir / f"scores_{DATASET_NAME}_{METHOD_FILE_TOKEN}_{split}.csv"
    enriched.to_csv(scores_csv, index=False)
    failures = failures_from_scores(enriched, split=split)
    failures_csv = outdir / f"failures_{DATASET_NAME}_{METHOD_FILE_TOKEN}_{split}.csv"
    failures.to_csv(failures_csv, index=False)

    n_failed = int((enriched["status"].astype(str) != "ok").sum())
    n_scored = n - n_failed
    n_positive = int((enriched["label"].astype(int) == 1).sum())
    n_negative = int((enriched["label"].astype(int) == 0).sum())

    run_meta = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "method": METHOD,
        "method_file_token": METHOD_FILE_TOKEN,
        "dataset": DATASET_NAME,
        "split": split,
        "pairs_csv": str(pairs_csv),
        "pairs_csv_sha256": sha256_file(pairs_csv),
        "scores_csv": str(scores_csv),
        "failures_csv": str(failures_csv),
        "limit": int(limit),
        "strict": bool(strict),
        "n_pairs": n,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "label_counts": {"n_positive": n_positive, "n_negative": n_negative},
        "timing": {**timing, "source": "deep_score_table_wall"},
        "model_provenance": frozen.provenance,
        "config": {
            "resolves_relative_paths": True,
            "regenerates_pairs": False,
            "trains_or_calibrates": False,
            "instantiates_optimizer_or_scheduler": False,
            "model_eval_mode": True,
            "no_grad_inference": True,
            "frozen": True,
            "batch_size": int(batch_size),
            "score_columns": list(enriched.columns),
        },
    }
    run_meta_json = outdir / f"run_{DATASET_NAME}_{METHOD_FILE_TOKEN}_{split}.meta.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return SplitResult(
        split=split,
        pairs_csv=pairs_csv,
        scores_csv=scores_csv,
        run_meta_json=run_meta_json,
        failures_csv=failures_csv,
        n_pairs=n,
        n_scored=n_scored,
        n_failed=n_failed,
        n_positive=n_positive,
        n_negative=n_negative,
        timing=timing,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    checkpoint: Path,
    manifest_dir: Path,
    outdir: Path,
    splits: list[str],
    limit: int,
    strict: bool,
    polyu_root: Optional[str],
    device: str,
    batch_size: int,
    num_workers: int,
    frozen: Optional[FrozenModel] = None,
) -> dict[str, Any]:
    manifest_dir = Path(manifest_dir)
    if not (manifest_dir / "manifest.csv").exists():
        raise PolyUCrossDeepRunError(f"PolyU Cross manifest.csv not found under {manifest_dir}")

    ensure_dir(outdir)
    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)
    split_csvs = dict(iter_pair_split_csvs(manifest_dir, splits))

    if frozen is None:
        frozen = load_frozen_model(Path(checkpoint), device_arg=device)

    results: list[SplitResult] = []
    for split in splits:
        pairs_csv = split_csvs[split]
        if not pairs_csv.exists():
            raise PolyUCrossDeepRunError(f"Missing PolyU Cross pairs CSV for split={split}: {pairs_csv}")
        print(f"[RUN] method={METHOD} split={split} pairs={pairs_csv.name}")
        result = run_split(
            frozen=frozen,
            split=split,
            pairs_csv=pairs_csv,
            root=resolved_root.root,
            outdir=outdir,
            limit=limit,
            strict=strict,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results.append(result)
        print(
            f"[DONE] split={split} scored={result.n_scored} failed={result.n_failed} "
            f"-> {result.scores_csv.name}"
        )

    latency_rows = [
        {
            "method": METHOD,
            "split": r.split,
            "n_pairs": r.n_pairs,
            "n_scored": r.n_scored,
            "n_failed": r.n_failed,
            "avg_ms_pair": r.timing.get("avg_ms_pair"),
            "total_ms": r.timing.get("total_ms"),
            "n_unique_images": r.timing.get("n_unique_images"),
            "device": frozen.device.type,
            "source": "deep_score_table_wall",
        }
        for r in results
    ]
    latency_csv = outdir / "latency_summary.csv"
    pd.DataFrame(latency_rows, columns=LATENCY_COLUMNS).to_csv(latency_csv, index=False)

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "method": METHOD,
        "method_file_token": METHOD_FILE_TOKEN,
        "repo_root": str(REPO_ROOT),
        "manifest_dir": str(manifest_dir),
        "outdir": str(outdir),
        "splits": list(splits),
        "limit": int(limit),
        "strict": bool(strict),
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "model_provenance": frozen.provenance,
        "runs": [
            {
                "split": r.split,
                "pairs_csv": str(r.pairs_csv),
                "pairs_csv_sha256": sha256_file(r.pairs_csv),
                "scores_csv": str(r.scores_csv),
                "run_meta_json": str(r.run_meta_json),
                "failures_csv": str(r.failures_csv),
                "n_pairs": r.n_pairs,
                "n_scored": r.n_scored,
                "n_failed": r.n_failed,
                "n_positive": r.n_positive,
                "n_negative": r.n_negative,
            }
            for r in results
        ],
        "latency_summary_csv": str(latency_csv),
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {
            "numpy": safe_pkg_version("numpy"),
            "pandas": safe_pkg_version("pandas"),
            "torch": safe_pkg_version("torch"),
        },
        "constraints": {
            "regenerated_pairs": False,
            "modified_manifest_or_pairs": False,
            "trained_or_finetuned": False,
            "calibrated_or_chose_thresholds": False,
            "ran_fusion": False,
            "modified_ui_or_api": False,
            "copied_biometric_images": False,
        },
        "notes": [
            "Frozen zero-shot transfer of the SD300 deep pair reranker to PolyU Cross.",
            "Checkpoint parameters are frozen (eval mode, no_grad, requires_grad=False); no fitting performed.",
            "Model + preprocessing reuse scripts/deep/score_fast_pair_ddp_splits.py unchanged.",
            "No thresholds selected and no final TAR/FAR claims.",
        ],
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {"manifest_json": manifest_json, "latency_csv": latency_csv, "results": results, "polyu_root": resolved_root, "frozen": frozen}


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen SD300 deep pair reranker -> zero-shot scoring on the existing PolyU Cross "
            "pair bundle. Reuses the canonical fast_ddp model and preprocessing. Does not train, "
            "calibrate, choose thresholds, regenerate pairs, or modify the manifest/pairs/UI/API."
        )
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--limit", type=int, default=0, help="If >0, score only the first N balanced pairs (smoke).")
    parser.add_argument("--strict", action="store_true", help="Fail (do not degrade) when images are missing.")
    parser.add_argument("--polyu_root", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = resolve_repo_path(args.checkpoint)
    manifest_dir = resolve_repo_path(args.data_dir)
    outdir = resolve_repo_path(args.outdir)
    splits = parse_csv_list(args.splits)
    if not splits:
        print("ERROR: --splits resolved to empty", file=sys.stderr)
        return 2

    try:
        summary = run(
            checkpoint=checkpoint,
            manifest_dir=manifest_dir,
            outdir=outdir,
            splits=splits,
            limit=int(args.limit),
            strict=bool(args.strict),
            polyu_root=str(args.polyu_root).strip() or None,
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
    except (PolyUCrossDeepRunError, PolyUCrossPairError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: list[SplitResult] = summary["results"]
    frozen: FrozenModel = summary["frozen"]
    total_scored = sum(r.n_scored for r in results)
    total_failed = sum(r.n_failed for r in results)
    print("\n=== PolyU Cross frozen deep-reranker run complete ===")
    print(f"Output dir     : {outdir}")
    print(f"Checkpoint     : {frozen.checkpoint}")
    print(f"Checkpoint SHA : {frozen.provenance.get('checkpoint_sha256')}")
    print(f"Device         : {frozen.device.type} | frozen={frozen.provenance.get('frozen')}")
    print(f"Run manifest   : {summary['manifest_json']}")
    print(f"Total scored   : {total_scored}")
    print(f"Total failed   : {total_failed}")
    for r in results:
        print(f"  - {r.split}: pairs={r.n_pairs} scored={r.n_scored} failed={r.n_failed} -> {r.scores_csv.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
