"""Deep-score saturation & ranking audit for PolyU Cross controls (Phase 4A.1C).

Read-only diagnostic. Determines whether the frozen SD300 deep model's
apparent CB->CB failure is a *true* cross-dataset representation failure or a
signal hidden by sigmoid saturation, by comparing AUC(probability) vs
AUC(raw logit) and quantifying saturation.

Logit availability
------------------
* CL->CB: the Phase 3A.2 score CSVs already store per-pair ``logit`` -> reused.
* CB->CB / CL->CL control protocols: Phase 4A.1B persisted only aggregate
  metrics; per-pair logits do not exist. Because the CB->CB probabilities
  saturated at 1.0, logits cannot be recovered from probabilities, so the raw
  logits are minimally recovered by re-running the *exact same* frozen
  checkpoint + inference path (``run_polyu_cross_deep_reranker``) on the
  *existing* 4A.1B control pair CSVs. Preprocessing/model behavior is unchanged;
  recovered per-pair logits are preserved under the audit output dir.

Nothing is trained, calibrated, thresholded, or fit. No probability mapping
(Platt/isotonic/temperature) is applied. TEST is never read. No pair file,
manifest, checkpoint, or existing score file is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
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
    load_polyu_cross_pairs,
    resolve_pairs_frame,
    resolve_polyu_cross_root,
)
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_DEEP_CLCB_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_deep_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_deep_saturation_audit_v0"
RUN_SCHEMA_VERSION = "polyu_cross_deep_saturation_audit_v0"
ALLOWED_SPLITS = ("train", "val")

# protocol_id -> short label
CONTROL_PROTOCOLS = {
    "contact_based_to_contact_based_same_session": "CB->CB same-session",
    "contact_based_to_contact_based_cross_session": "CB->CB cross-session",
    "contactless_to_contactless_same_session": "CL->CL same-session",
    "contactless_to_contactless_cross_session": "CL->CL cross-session",
}
CLCB_PROTOCOL = "contactless_to_contact_based"
CLCB_LABEL = "CL->CB (existing)"
CB_CB_PROTOCOLS = ("contact_based_to_contact_based_same_session", "contact_based_to_contact_based_cross_session")


class SaturationAuditError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    labels, values = labels[finite], values[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, values))  # returns 0.5 for constant values
    except ValueError:
        return float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation (ties-averaged) without scipy: Pearson of average ranks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _grp_stats(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0, float(np.median(v))


def saturation_stats(
    labels: np.ndarray, score: np.ndarray, logit: np.ndarray, *, protocol: str, label_short: str, split: str, source: str
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    logit = np.asarray(logit, dtype=float)
    finite_score = np.isfinite(score)
    finite_logit = np.isfinite(logit)
    scored = int((finite_score & finite_logit).sum())

    gmask = labels == 1
    imask = labels == 0
    g_s = _grp_stats(score[gmask]); i_s = _grp_stats(score[imask])
    g_l = _grp_stats(logit[gmask]); i_l = _grp_stats(logit[imask])

    s_fin = score[finite_score]
    return {
        "protocol": protocol,
        "protocol_short": label_short,
        "split": split,
        "logit_source": source,
        "pair_count": int(len(labels)),
        "genuine_count": int(gmask.sum()),
        "impostor_count": int(imask.sum()),
        "scored_count": scored,
        "failed_count": int(len(labels) - scored),
        "auc_score": _auc(labels, score),
        "auc_logit": _auc(labels, logit),
        "auc_logit_minus_score": _auc(labels, logit) - _auc(labels, score),
        "spearman_score_logit": _spearman(score, logit),
        "genuine_score_mean": g_s[0], "genuine_score_std": g_s[1], "genuine_score_median": g_s[2],
        "impostor_score_mean": i_s[0], "impostor_score_std": i_s[1], "impostor_score_median": i_s[2],
        "genuine_logit_mean": g_l[0], "genuine_logit_std": g_l[1], "genuine_logit_median": g_l[2],
        "impostor_logit_mean": i_l[0], "impostor_logit_std": i_l[1], "impostor_logit_median": i_l[2],
        "frac_score_eq_0": float(np.mean(s_fin == 0.0)) if s_fin.size else float("nan"),
        "frac_score_eq_1": float(np.mean(s_fin == 1.0)) if s_fin.size else float("nan"),
        "frac_score_within_1e6_of_0": float(np.mean(np.abs(s_fin - 0.0) <= 1e-6)) if s_fin.size else float("nan"),
        "frac_score_within_1e6_of_1": float(np.mean(np.abs(s_fin - 1.0) <= 1e-6)) if s_fin.size else float("nan"),
        "n_unique_score": int(np.unique(s_fin).size),
        "n_unique_logit": int(np.unique(logit[finite_logit]).size),
        "min_score": float(np.min(s_fin)) if s_fin.size else float("nan"),
        "max_score": float(np.max(s_fin)) if s_fin.size else float("nan"),
        "min_logit": float(np.min(logit[finite_logit])) if finite_logit.any() else float("nan"),
        "max_logit": float(np.max(logit[finite_logit])) if finite_logit.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Score sources
# ---------------------------------------------------------------------------
def recover_control_logits(frozen: Any, pairs_csv: Path, root: Optional[Path]) -> pd.DataFrame:
    """Re-run the frozen deep model on an existing control pair CSV to recover
    per-pair logits (probabilities saturated -> logits unrecoverable otherwise)."""
    from pipelines.benchmark.run_polyu_cross_deep_reranker import score_resolved_pairs

    pairs = load_polyu_cross_pairs(pairs_csv)
    resolved = resolve_pairs_frame(pairs, root).reset_index(drop=True)
    missing = resolved[(~resolved["path_a_exists"]) | (~resolved["path_b_exists"])]
    if not missing.empty:
        raise SaturationAuditError(f"{len(missing)} control pair(s) reference missing images in {pairs_csv.name}")
    logits, probs, _timing = score_resolved_pairs(frozen, resolved)
    return pd.DataFrame(
        {
            "pair_id": resolved["pair_id"].astype(str).values,
            "label": resolved["label"].astype(int).values,
            "score": np.asarray(probs, dtype=float),
            "logit": np.asarray(logits, dtype=float),
        }
    )


def load_clcb_scores(deep_clcb_dir: Path, split: str) -> pd.DataFrame:
    path = Path(deep_clcb_dir) / f"scores_polyu_cross_deep_pair_reranker_{split}.csv"
    if not path.exists():
        raise SaturationAuditError(f"Existing CL->CB deep scores not found: {path}")
    df = pd.read_csv(path)
    if not {"label", "score", "logit"}.issubset(df.columns):
        raise SaturationAuditError(f"CL->CB scores {path} missing label/score/logit columns")
    return df[["pair_id", "label", "score", "logit"]].copy()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_cbcb(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the CB->CB result from VAL AUC(logit) vs AUC(score)."""
    val = [r for r in rows if r["protocol"] in CB_CB_PROTOCOLS and r["split"] == "val"]
    best_logit = max((r["auc_logit"] for r in val), default=float("nan"))
    max_delta = max((r["auc_logit_minus_score"] for r in val), default=float("nan"))

    def _above_chance(a: float) -> float:
        return abs(a - 0.5)

    material = np.isfinite(max_delta) and max_delta > 0.03
    clearly_above = np.isfinite(best_logit) and best_logit >= 0.60
    approx_chance = np.isfinite(best_logit) and _above_chance(best_logit) < 0.05

    if material and clearly_above:
        label = "A. REPRESENTATION_SIGNAL_HIDDEN_BY_SATURATION"
    elif approx_chance and not clearly_above:
        label = "B. TRUE_CROSS_DATASET_REPRESENTATION_FAILURE"
    else:
        label = "C. MIXED_OR_INCONCLUSIVE"
    return {
        "classification": label,
        "val_cbcb_best_auc_logit": float(best_logit),
        "val_cbcb_max_auc_logit_minus_score": float(max_delta),
        "criteria": {
            "material_delta_gt_0.03": bool(material),
            "logit_auc_ge_0.60": bool(clearly_above),
            "logit_auc_within_0.05_of_chance": bool(approx_chance),
        },
        "per_protocol_val": {r["protocol"]: {"auc_score": r["auc_score"], "auc_logit": r["auc_logit"]} for r in val},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    manifest_dir: Path,
    controls_dir: Path,
    deep_clcb_dir: Path,
    checkpoint: Path,
    outdir: Path,
    polyu_root: Optional[str],
    frozen: Any = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recovered_dir = outdir / "recovered_scores"
    recovered_dir.mkdir(parents=True, exist_ok=True)

    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)

    # Load frozen model only for the control protocols that need logit recovery.
    rerun_inference = False
    deep_provenance: dict[str, Any] = {}
    if frozen is None:
        from pipelines.benchmark.run_polyu_cross_deep_reranker import load_frozen_model

        frozen = load_frozen_model(Path(checkpoint), device_arg="auto")
    deep_provenance = getattr(frozen, "provenance", {})

    rows: list[dict[str, Any]] = []
    files_read: list[str] = []

    for protocol, short in CONTROL_PROTOCOLS.items():
        for split in ALLOWED_SPLITS:
            pairs_csv = Path(controls_dir) / "pairs" / f"pairs_{protocol}_{split}.csv"
            if not pairs_csv.exists():
                raise SaturationAuditError(f"Missing control pair CSV (need it read-only): {pairs_csv}")
            files_read.append(str(pairs_csv))
            recovered = recover_control_logits(frozen, pairs_csv, resolved_root.root)
            rerun_inference = True
            recovered.to_csv(recovered_dir / f"recovered_{protocol}_{split}.csv", index=False)
            rows.append(
                saturation_stats(
                    recovered["label"].to_numpy(), recovered["score"].to_numpy(), recovered["logit"].to_numpy(),
                    protocol=protocol, label_short=short, split=split, source="recovered_rerun_same_checkpoint",
                )
            )

    for split in ALLOWED_SPLITS:
        clcb = load_clcb_scores(deep_clcb_dir, split)
        files_read.append(str(Path(deep_clcb_dir) / f"scores_polyu_cross_deep_pair_reranker_{split}.csv"))
        rows.append(
            saturation_stats(
                clcb["label"].to_numpy(), clcb["score"].to_numpy(), clcb["logit"].to_numpy(),
                protocol=CLCB_PROTOCOL, label_short=CLCB_LABEL, split=split, source="reused_existing_csv",
            )
        )

    audit = pd.DataFrame(rows).sort_values(["protocol", "split"], kind="mergesort").reset_index(drop=True)
    audit_csv = outdir / "saturation_audit.csv"
    audit.to_csv(audit_csv, index=False)

    val = audit[audit["split"] == "val"][
        ["protocol_short", "protocol", "auc_score", "auc_logit", "auc_logit_minus_score", "n_unique_score", "n_unique_logit", "frac_score_within_1e6_of_1"]
    ].reset_index(drop=True)
    val_csv = outdir / "val_comparison.csv"
    val.to_csv(val_csv, index=False)

    classification = classify_cbcb(rows)

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "splits_audited": list(ALLOWED_SPLITS),
        "protocol": "TRAIN and VAL only; TEST never read.",
        "inference_rerun": bool(rerun_inference),
        "inference_rerun_reason": (
            "Per-pair logits were not persisted for the CB->CB / CL->CL control protocols in 4A.1B, and "
            "CB->CB probabilities saturated at 1.0 (logits unrecoverable from probs); recovered via the exact "
            "same frozen checkpoint + inference path on the existing 4A.1B control pair CSVs."
        ),
        "logit_sources": {
            "control_protocols": "recovered_rerun_same_checkpoint (run_polyu_cross_deep_reranker.score_resolved_pairs)",
            "contactless_to_contact_based": "reused existing Phase 3A.2 score CSVs (logit already present)",
        },
        "files_read": files_read,
        "outputs": {"saturation_audit_csv": str(audit_csv), "val_comparison_csv": str(val_csv), "recovered_scores_dir": str(recovered_dir)},
        "deep_model_provenance": deep_provenance,
        "cbcb_classification": classification,
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "constraints": {
            "reran_image_inference": bool(rerun_inference),
            "trained_or_finetuned": False,
            "calibrated_or_fit_mapping": False,
            "selected_thresholds": False,
            "evaluated_test": False,
            "modified_pairs_manifest_checkpoint_or_scores": False,
        },
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {"audit": audit, "val": val, "classification": classification, "manifest_json": manifest_json, "outdir": outdir, "rerun": rerun_inference}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PolyU Cross frozen-deep saturation & ranking audit (read-only, TRAIN/VAL).")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--deep_clcb_dir", type=str, default=DEFAULT_DEEP_CLCB_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    return p


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            manifest_dir=_resolve_repo_path(args.data_dir),
            controls_dir=_resolve_repo_path(args.controls_dir),
            deep_clcb_dir=_resolve_repo_path(args.deep_clcb_dir),
            checkpoint=_resolve_repo_path(args.checkpoint),
            outdir=_resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
        )
    except SaturationAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n=== PolyU Cross deep saturation audit complete ===")
    print(f"Output dir       : {result['outdir']}")
    print(f"Inference rerun  : {result['rerun']} (control protocols only; CL->CB reused)")
    print("\nVAL: AUC(score) vs AUC(logit):")
    print(result["val"][["protocol_short", "auc_score", "auc_logit", "auc_logit_minus_score", "n_unique_score", "n_unique_logit"]].to_string(index=False))
    print(f"\nCB->CB classification: {result['classification']['classification']}")
    print(f"  VAL CB->CB best AUC(logit)={result['classification']['val_cbcb_best_auc_logit']:.4f} "
          f"max delta={result['classification']['val_cbcb_max_auc_logit_minus_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
