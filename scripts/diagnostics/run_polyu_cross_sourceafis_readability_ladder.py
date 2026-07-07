"""SourceAFIS contactless-readability ladder for PolyU Cross (Phase 4A.2A).

Isolates *one deterministic contactless image transformation at a time* and
measures, with real SourceAFIS, whether it (a) makes contactless fingerprints
readable (CL->CL) and (b) improves cross-modality matching (CL->CB).

Variants (contactless side only; contact-based images are never modified):
  P0_raw                  - identity (reuses existing SourceAFIS CL->CB scores)
  P1_invert               - 255 - image
  P2_robust_intensity_norm- per-image p05->0, p95->255, clip (degenerate-safe)
  P3_clahe                - CLAHE (clipLimit=2.0, tileGridSize=8x8)
  P4_roi_crop_pad         - foreground crop -> isotropic resize -> pad to 512x512

Transforms are pure/deterministic and applied to *contactless* images only.
Transformed images live in an ephemeral scratch dir cleaned up after scoring;
nothing is persisted under data/raw or data/manifests, and raw images are
read-only. No training, calibration, thresholds, TEST usage, deep model,
transform combination, SourceAFIS-parameter change, or sidecar-API change.

Protocols (TRAIN/VAL only): CL->CL same-session, CL->CL cross-session, and the
canonical CL->CB bundle. Pairs are reused, never regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    load_polyu_cross_pairs,
    resolve_pairs_frame,
    resolve_polyu_cross_root,
)
from src.fpbench.deep.transforms import crop_foreground, resize_with_padding
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
CONTACTLESS = "contactless_2d"
CONTACT = "contact_based_2d"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_SOURCEAFIS_CLCB_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0_sourceafis_real"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_sourceafis_readability_ladder_v0"
RUN_SCHEMA_VERSION = "polyu_cross_sourceafis_readability_ladder_v0"
ALLOWED_SPLITS = ("train", "val")

# Documented transform parameters.
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
P2_LO_PCT, P2_HI_PCT = 5.0, 95.0
P4_CANVAS = 512
P4_CROP_MARGIN = 16
P4_PAD_FILL = 255
CLCB_TRAIN_SUBSET_POS = 300  # deterministic dev subset for the large CL->CB train split
CLCB_TRAIN_SUBSET_NEG = 900

CLCL_SAME = "contactless_to_contactless_same_session"
CLCL_CROSS = "contactless_to_contactless_cross_session"
CLCB = "contactless_to_contact_based"


class ReadabilityLadderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Deterministic pure transforms (uint8 grayscale -> uint8 grayscale)
# ---------------------------------------------------------------------------
def p0_raw(img: np.ndarray) -> np.ndarray:
    return img


def p1_invert(img: np.ndarray) -> np.ndarray:
    return (255 - img.astype(np.uint8)).astype(np.uint8)


def p2_robust_intensity_norm(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [P2_LO_PCT, P2_HI_PCT])
    rng = float(hi) - float(lo)
    if rng < 1.0:  # degenerate percentile range -> safe identity passthrough
        return img.astype(np.uint8)
    out = (img.astype(np.float64) - float(lo)) * (255.0 / rng)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def p3_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(img.astype(np.uint8))


def p4_roi_crop_pad(img: np.ndarray) -> np.ndarray:
    cropped = crop_foreground(img.astype(np.uint8), margin=P4_CROP_MARGIN)
    if cropped.size == 0:
        cropped = img.astype(np.uint8)
    return resize_with_padding(cropped, size=P4_CANVAS, fill=P4_PAD_FILL).astype(np.uint8)


VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "P0_raw": p0_raw,
    "P1_invert": p1_invert,
    "P2_robust_intensity_norm": p2_robust_intensity_norm,
    "P3_clahe": p3_clahe,
    "P4_roi_crop_pad": p4_roi_crop_pad,
}
VARIANT_ORDER = list(VARIANTS.keys())

VARIANT_PARAMS = {
    "P0_raw": {"op": "identity"},
    "P1_invert": {"op": "255 - image"},
    "P2_robust_intensity_norm": {"op": "per-image percentile stretch", "lo_pct": P2_LO_PCT, "hi_pct": P2_HI_PCT,
                                  "map": "p05->0, p95->255, clip", "degenerate": "range<1 -> identity", "uses_gallery_stats": False},
    "P3_clahe": {"op": "CLAHE", "clipLimit": CLAHE_CLIP, "tileGridSize": list(CLAHE_GRID)},
    "P4_roi_crop_pad": {"op": "foreground crop -> isotropic resize -> pad", "crop": "deep.transforms.crop_foreground",
                         "crop_margin": P4_CROP_MARGIN, "resize": "isotropic aspect-preserving (deep.transforms.resize_with_padding)",
                         "canvas": [P4_CANVAS, P4_CANVAS], "pad_fill": P4_PAD_FILL, "stretch": False},
}


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ReadabilityLadderError(f"Failed to read image: {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Transformed-path construction (contactless side only)
# ---------------------------------------------------------------------------
def build_transformed_resolved(resolved: pd.DataFrame, variant: str, tmp_dir: Path) -> pd.DataFrame:
    """Return a resolved frame whose contactless-side paths point at transformed
    images written under ``tmp_dir``. Contact-based paths are left untouched.

    ``P0_raw`` performs no transform and keeps original raw paths (read-only).
    """
    transform = VARIANTS[variant]
    is_p0 = variant == "P0_raw"
    cache: dict[str, str] = {}
    out = resolved.copy()

    def _side_path(modality: str, sample_uid: str, original_resolved: str) -> str:
        if modality != CONTACTLESS or is_p0:
            return original_resolved  # contact unchanged; P0 contactless uses raw
        if sample_uid not in cache:
            dst = tmp_dir / f"{sample_uid}.bmp"
            arr = transform(load_gray(original_resolved))
            if not cv2.imwrite(str(dst), arr):
                raise ReadabilityLadderError(f"Failed to write transformed image: {dst}")
            cache[sample_uid] = str(dst)
        return cache[sample_uid]

    new_a, new_b = [], []
    for _, row in out.iterrows():
        new_a.append(_side_path(str(row["modality_a"]), str(row["sample_uid_a"]), str(row["resolved_path_a"])))
        new_b.append(_side_path(str(row["modality_b"]), str(row["sample_uid_b"]), str(row["resolved_path_b"])))
    out["resolved_path_a"] = new_a
    out["resolved_path_b"] = new_b
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(values)
    labels, values = labels[finite], values[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, values))
    except ValueError:
        return float("nan")


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    gmask = labels == 1
    imask = labels == 0

    def _stats(a: np.ndarray) -> tuple[float, float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.mean(a)), float(np.median(a)), float(np.std(a, ddof=1)) if a.size > 1 else 0.0

    g_mean, g_med, g_std = _stats(scores[gmask])
    i_mean, i_med, i_std = _stats(scores[imask])
    nz = finite & (scores > 0)

    def _frac(mask: np.ndarray) -> float:
        n = int(mask.sum())
        return float(np.sum(nz & mask) / n) if n else float("nan")

    return {
        "pair_count": int(len(labels)),
        "genuine_count": int(gmask.sum()),
        "impostor_count": int(imask.sum()),
        "scored_count": int(finite.sum()),
        "failed_count": int((~finite).sum()),
        "roc_auc": _auc(labels, scores),
        "genuine_score_mean": g_mean, "genuine_score_median": g_med, "genuine_score_std": g_std,
        "impostor_score_mean": i_mean, "impostor_score_median": i_med, "impostor_score_std": i_std,
        "fraction_nonzero_all": _frac(np.ones_like(labels, dtype=bool)),
        "fraction_nonzero_genuine": _frac(gmask),
        "fraction_nonzero_impostor": _frac(imask),
    }


# ---------------------------------------------------------------------------
# Pair sources
# ---------------------------------------------------------------------------
def load_protocol_pairs(protocol: str, split: str, *, manifest_dir: Path, controls_dir: Path) -> pd.DataFrame:
    if protocol in (CLCL_SAME, CLCL_CROSS):
        path = Path(controls_dir) / "pairs" / f"pairs_{protocol}_{split}.csv"
        if not path.exists():
            raise ReadabilityLadderError(f"Missing control pair CSV (reuse, read-only): {path}")
        return load_polyu_cross_pairs(path)
    if protocol == CLCB:
        path = Path(manifest_dir) / f"pairs_{split}.csv"
        pairs = load_polyu_cross_pairs(path)
        if split == "train":
            pos = pairs[pairs["label"] == 1].sort_values("pair_id", kind="mergesort").head(CLCB_TRAIN_SUBSET_POS)
            neg = pairs[pairs["label"] == 0].sort_values("pair_id", kind="mergesort").head(CLCB_TRAIN_SUBSET_NEG)
            pairs = pd.concat([pos, neg], axis=0).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
        return pairs
    raise ReadabilityLadderError(f"Unknown protocol {protocol}")


def load_existing_clcb_sourceafis(clcb_dir: Path, split: str, pair_ids: set[str]) -> Optional[pd.DataFrame]:
    path = Path(clcb_dir) / f"scores_polyu_cross_sourceafis_open_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"pair_id", "label", "score"}.issubset(df.columns):
        return None
    df = df[df["pair_id"].astype(str).isin(pair_ids)].copy()
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    manifest_dir: Path,
    controls_dir: Path,
    sourceafis_clcb_dir: Path,
    outdir: Path,
    polyu_root: Optional[str],
    variants: list[str],
    protocols: list[str],
    splits: list[str],
    scratch_root: Optional[Path] = None,
) -> dict[str, Any]:
    from pipelines.benchmark.run_polyu_cross_zero_shot import score_split_sourceafis

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work_dir = outdir / "_work"  # persistent SourceAFIS template cache (gitignored, cleaned at end)
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(scratch_root) if scratch_root else Path(tempfile.mkdtemp(prefix="polyu_readability_"))
    scratch_root.mkdir(parents=True, exist_ok=True)

    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)
    rows: list[dict[str, Any]] = []
    scores_dir = outdir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    try:
        for variant in variants:
            for protocol in protocols:
                for split in splits:
                    pairs = load_protocol_pairs(protocol, split, manifest_dir=manifest_dir, controls_dir=controls_dir)
                    labels = pairs["label"].astype(int).to_numpy()
                    base = {"variant": variant, "protocol": protocol, "split": split}

                    # P0 + CL->CB: reuse existing raw SourceAFIS scores (exact pair identity).
                    if variant == "P0_raw" and protocol == CLCB:
                        reused = load_existing_clcb_sourceafis(sourceafis_clcb_dir, split, set(pairs["pair_id"].astype(str)))
                        if reused is not None and len(reused) == len(pairs):
                            order = {pid: i for i, pid in enumerate(pairs["pair_id"].astype(str))}
                            reused = reused.assign(_o=reused["pair_id"].astype(str).map(order)).sort_values("_o")
                            m = compute_metrics(reused["label"].astype(int).to_numpy(), pd.to_numeric(reused["score"], errors="coerce").to_numpy())
                            rows.append({**base, "score_source": "reused_existing_raw_sourceafis", **m})
                            continue

                    resolved = resolve_pairs_frame(pairs, resolved_root.root)
                    tmp_dir = scratch_root / f"{variant}_{protocol}_{split}"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        transformed = build_transformed_resolved(resolved, variant, tmp_dir)
                        enriched, _timing = score_split_sourceafis(split=split, resolved_df=transformed, work_dir=work_dir)
                        scores = pd.to_numeric(enriched["score"], errors="coerce").to_numpy(dtype=float)
                        m = compute_metrics(labels, scores)
                        rows.append({**base, "score_source": "scored_transformed", **m})
                        # Optional per-variant/protocol score file (identity + score), for reproducibility.
                        keep = ["pair_id", "label", "score", "status"]
                        enriched[[c for c in keep if c in enriched.columns]].to_csv(
                            scores_dir / f"scores_{variant}_{protocol}_{split}.csv", index=False
                        )
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)

        metrics = pd.DataFrame(rows)
        metrics["_v"] = metrics["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
        metrics = metrics.sort_values(["_v", "protocol", "split"], kind="mergesort").drop(columns="_v").reset_index(drop=True)
        ladder_csv = outdir / "ladder_metrics.csv"
        metrics.to_csv(ladder_csv, index=False)

        val_csv = outdir / "val_comparison.csv"
        val_tbl = _val_comparison(metrics)
        val_tbl.to_csv(val_csv, index=False)

        manifest = _build_manifest(
            manifest_dir=manifest_dir, controls_dir=controls_dir, sourceafis_clcb_dir=sourceafis_clcb_dir,
            outdir=outdir, resolved_root=resolved_root, variants=variants, protocols=protocols, splits=splits,
            ladder_csv=ladder_csv, val_csv=val_csv,
        )
        manifest_json = outdir / "run_manifest.json"
        manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)  # ephemeral transformed images cleaned up
        shutil.rmtree(work_dir, ignore_errors=True)  # ephemeral template cache cleaned up

    return {"metrics": metrics, "val": val_tbl, "ladder_csv": ladder_csv, "val_csv": val_csv, "manifest_json": manifest_json, "outdir": outdir}


def _val_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    val = metrics[metrics["split"] == "val"]

    def pick(variant: str, protocol: str, col: str) -> float:
        sub = val[(val["variant"] == variant) & (val["protocol"] == protocol)]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    rows = []
    for variant in VARIANT_ORDER:
        if variant not in set(metrics["variant"]):
            continue
        rows.append(
            {
                "variant": variant,
                "clcl_same_auc": pick(variant, CLCL_SAME, "roc_auc"),
                "clcl_cross_auc": pick(variant, CLCL_CROSS, "roc_auc"),
                "clcb_auc": pick(variant, CLCB, "roc_auc"),
                "clcl_same_nonzero_genuine": pick(variant, CLCL_SAME, "fraction_nonzero_genuine"),
                "clcl_cross_nonzero_genuine": pick(variant, CLCL_CROSS, "fraction_nonzero_genuine"),
                "clcb_nonzero_genuine": pick(variant, CLCB, "fraction_nonzero_genuine"),
            }
        )
    return pd.DataFrame(rows)


def _build_manifest(**kw: Any) -> dict[str, Any]:
    resolved_root = kw["resolved_root"]
    manifest_dir = kw["manifest_dir"]
    pairs_shas = {}
    for split in kw["splits"]:
        p = Path(manifest_dir) / f"pairs_{split}.csv"
        if p.exists():
            pairs_shas[f"canonical_pairs_{split}"] = sha256_file(p)
    pairs_shas["manifest"] = sha256_file(Path(manifest_dir) / "manifest.csv")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "protocol": "TRAIN/VAL only; TEST never read. One transform at a time; contactless side only.",
        "variants": kw["variants"],
        "variant_parameters": VARIANT_PARAMS,
        "protocols": kw["protocols"],
        "splits": kw["splits"],
        "clcb_train_subset": {"pos": CLCB_TRAIN_SUBSET_POS, "neg": CLCB_TRAIN_SUBSET_NEG,
                               "note": "Deterministic dev subset of the large canonical CL->CB train split (val uses full bundle)."},
        "polyu_root": {"path": str(resolved_root.root) if resolved_root.root else None, "source": resolved_root.source},
        "reused_inputs": {
            "clcl_pairs": "artifacts/.../polyu_cross_modality_controls_v0/pairs (4A.1B, read-only)",
            "clcb_pairs": "canonical data/manifests/polyu_cross/pairs_<split>.csv (read-only, not regenerated)",
            "clcb_P0_scores": "reused existing raw SourceAFIS CL->CB scores where pair identity matched",
        },
        "readability_diagnostic": "fraction_nonzero (SourceAFIS raw score > 0). Sidecar extract API exposes no minutiae count; not modified.",
        "transformed_image_storage": "ephemeral scratch (bmp), cleaned up after scoring; no persistence under data/raw or data/manifests",
        "sha256": pairs_shas,
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "constraints": {
            "trained_or_finetuned": False, "calibrated_scores": False, "selected_thresholds": False,
            "evaluated_test": False, "modified_manifest_or_pairs": False, "modified_contact_images": False,
            "modified_sourceafis_params": False, "modified_sidecar_api": False, "ran_deep_model": False,
            "combined_transforms": False, "persisted_transformed_images_under_data": False,
        },
        "outputs": {"ladder_metrics_csv": str(kw["ladder_csv"]), "val_comparison_csv": str(kw["val_csv"])},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SourceAFIS contactless-readability ladder for PolyU Cross (diagnostics, TRAIN/VAL).")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--sourceafis_clcb_dir", type=str, default=DEFAULT_SOURCEAFIS_CLCB_DIR)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    p.add_argument("--protocols", type=str, default=",".join([CLCL_SAME, CLCL_CROSS, CLCB]))
    p.add_argument("--splits", type=str, default="train,val")
    return p


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad:
        print(f"ERROR: refusing non-development splits {bad}; allowed {list(ALLOWED_SPLITS)}", file=sys.stderr)
        return 2
    try:
        result = run(
            manifest_dir=_resolve_repo_path(args.data_dir),
            controls_dir=_resolve_repo_path(args.controls_dir),
            sourceafis_clcb_dir=_resolve_repo_path(args.sourceafis_clcb_dir),
            outdir=_resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            variants=variants, protocols=protocols, splits=splits,
        )
    except ReadabilityLadderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n=== PolyU Cross SourceAFIS readability ladder complete ===")
    print(f"Output dir : {result['outdir']}")
    print("\nVAL comparison:")
    print(result["val"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
