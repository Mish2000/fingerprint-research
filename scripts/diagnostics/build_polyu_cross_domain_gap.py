"""PolyU Cross domain-gap audit (Phase 4A.1) - diagnostics only.

Quantifies the acquisition-domain gap between contactless_2d and
contact_based_2d samples of the *existing* PolyU Cross pair bundle. It computes
deterministic per-image domain features for every TRAIN and VAL image, then
summarizes modality-level statistics, standardized effect sizes, and
paired-genuine cross-modality feature gaps.

Strictly diagnostic. It does not train, fine-tune, calibrate, choose
thresholds, regenerate pairs, edit the manifest/pairs, touch UI/API, evaluate
TEST, or copy biometric images. TEST pairs are never read.

Feature provenance / reuse
--------------------------
* Canonical quality features reuse ``src.fpbench.universal.quality.extract_image_quality``
  (mean, std, contrast_proxy, Otsu foreground_ratio, Laplacian sharpness,
  edge_density).
* Foreground bbox occupancy reuses ``src.fpbench.deep.transforms.foreground_bbox``.
* Percentiles, entropy, and local contrast are standard deterministic image
  statistics.
* ``orientation_dominant_deg`` / ``orientation_coherence`` use a minimal,
  validated structure-tensor (gradient-covariance) estimator restricted to the
  foreground bounding box. This is the standard reliable orientation-coherence
  method, not an invented per-block ridge estimator.

Omitted features (documented in the run manifest)
-------------------------------------------------
* ridge_frequency: no existing reliable estimator; a robust cross-modality
  ridge-frequency estimator would need validation beyond this diagnostic and
  risks unreliable values on contactless fingerphotos.
* polarity_statistic: the only polarity logic lives inside the contact-tuned
  minutiae extraction pipeline (``_binarize_ridge_polarity_candidates`` /
  ``_needs_inverse_polarity_fallback``); it is not a standalone reliable
  per-image statistic.

Outputs (under ``--outdir``)
----------------------------
* ``image_domain_features.csv``            - one row per unique TRAIN/VAL image.
* ``domain_gap_summary.csv``               - per-feature per-modality stats + Cohen's d.
* ``paired_genuine_feature_gaps_by_session.csv`` - genuine cross-modality gaps.
* ``run_manifest.json``                    - provenance manifest.
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

import cv2

from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    PolyUCrossPairError,
    iter_pair_split_csvs,
    load_polyu_cross_pairs,
    resolve_pair_image_path,
    resolve_polyu_cross_root,
)
from src.fpbench.deep.transforms import foreground_bbox
from src.fpbench.universal.quality import extract_image_quality
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_domain_gap_v0"
# Development/diagnostic protocol: TRAIN and VAL only. TEST must never be read.
ALLOWED_SPLITS = ("train", "val")
RUN_SCHEMA_VERSION = "polyu_cross_domain_gap_v0"

CONTACTLESS = "contactless_2d"
CONTACT = "contact_based_2d"

# Per-image metadata columns (deterministic identity).
META_COLUMNS = ["sample_uid", "split", "modality", "session_id", "finger_unit_id", "subject_id", "path"]

# Numeric features summarized for the modality gap. Circular
# orientation_dominant_deg is intentionally excluded from mean/std/effect-size.
NUMERIC_FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "gray_mean",
    "gray_std",
    "gray_p05",
    "gray_p25",
    "gray_p50",
    "gray_p75",
    "gray_p95",
    "entropy",
    "contrast_proxy",
    "local_contrast",
    "foreground_ratio",
    "foreground_bbox_occupancy",
    "sharpness_laplacian_var",
    "edge_density",
    "orientation_coherence",
]

# Subset that must be finite for every image (validation requirement).
REQUIRED_BASIC_FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "gray_mean",
    "gray_std",
    "gray_p05",
    "gray_p25",
    "gray_p50",
    "gray_p75",
    "gray_p95",
    "entropy",
    "foreground_ratio",
    "sharpness_laplacian_var",
    "local_contrast",
]

OMITTED_FEATURES = {
    "ridge_frequency": (
        "No existing reliable estimator in the repository. A robust cross-modality "
        "ridge-frequency estimator would require validation beyond this diagnostic and "
        "risks unreliable values on contactless fingerphotos, so it is omitted rather "
        "than invented."
    ),
    "polarity_statistic": (
        "The only polarity logic is embedded in the contact-tuned minutiae extraction "
        "pipeline (minutiae_matcher._binarize_ridge_polarity_candidates / "
        "_needs_inverse_polarity_fallback) and is not a standalone reliable per-image "
        "statistic; it is omitted rather than approximated unreliably."
    ),
}


class DomainGapError(RuntimeError):
    """Raised for unrecoverable audit setup/protocol failures."""


# ---------------------------------------------------------------------------
# Image collection (TRAIN/VAL only)
# ---------------------------------------------------------------------------
def collect_bundle_images(manifest_dir: Path, splits: list[str], root: Optional[Path]) -> pd.DataFrame:
    """Collect the unique set of images referenced by the given splits' pairs.

    Emits one record per (sample_uid) with its modality/session/finger_unit and
    resolved path. Guards against TEST leakage and metadata inconsistency.
    """

    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad:
        raise DomainGapError(f"Refusing to audit non-development splits {bad}; allowed: {list(ALLOWED_SPLITS)}")

    records: list[dict[str, Any]] = []
    for split, pairs_csv in iter_pair_split_csvs(manifest_dir, splits):
        if not pairs_csv.exists():
            raise DomainGapError(f"Missing PolyU Cross pairs CSV for split={split}: {pairs_csv}")
        pairs = load_polyu_cross_pairs(pairs_csv)
        for side, uid_c, mod_c, ses_c, fu_c, sub_c, path_c in (
            ("a", "sample_uid_a", "modality_a", "session_a", "finger_unit_a", "subject_a", "path_a"),
            ("b", "sample_uid_b", "modality_b", "session_b", "finger_unit_b", "subject_b", "path_b"),
        ):
            sub = pd.DataFrame(
                {
                    "sample_uid": pairs[uid_c].astype(str).values,
                    "split": str(split),
                    "modality": pairs[mod_c].astype(str).values if mod_c in pairs else "",
                    "session_id": pairs[ses_c].astype(str).values if ses_c in pairs else "",
                    "finger_unit_id": pairs[fu_c].astype(str).values if fu_c in pairs else "",
                    "subject_id": pairs[sub_c].astype(str).values if sub_c in pairs else "",
                    "path": pairs[path_c].astype(str).values,
                }
            )
            records.append(sub)

    images = pd.concat(records, axis=0, ignore_index=True)
    # Deterministic dedupe by sample_uid.
    images = images.sort_values(["sample_uid"], kind="mergesort").drop_duplicates("sample_uid", keep="first")

    # Consistency guard: a sample_uid must not carry conflicting modality/split.
    for key in ("modality", "split"):
        counts = images.groupby("sample_uid")[key].nunique()
        conflicts = counts[counts > 1]
        if not conflicts.empty:
            raise DomainGapError(f"sample_uid maps to multiple {key} values (bundle inconsistency): {list(conflicts.index[:5])}")

    # Resolve + require existence.
    resolved: list[str] = []
    missing: list[str] = []
    for raw in images["path"].astype(str):
        p = resolve_pair_image_path(raw, root)
        resolved.append(str(p))
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise DomainGapError(f"{len(missing)} image path(s) did not resolve to an existing file. First: {missing[0]!r}")
    images = images.copy()
    images["resolved_path"] = resolved
    return images.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deterministic per-image features
# ---------------------------------------------------------------------------
def _shannon_entropy(img: np.ndarray) -> float:
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _local_contrast(img: np.ndarray, ksize: int = 15) -> float:
    """Mean local standard deviation (deterministic local-contrast metric)."""
    f = img.astype(np.float64)
    mean = cv2.boxFilter(f, ddepth=-1, ksize=(ksize, ksize), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(f * f, ddepth=-1, ksize=(ksize, ksize), normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return float(np.mean(np.sqrt(var)))


def structure_tensor_orientation(img: np.ndarray) -> tuple[float, float]:
    """Dominant orientation (deg, [0,180)) and coherence ([0,1]) via a global
    structure tensor over the foreground bounding box.

    Minimal, validated, deterministic gradient-covariance method.
    """
    x0, y0, x1, y1 = foreground_bbox(img, margin=0)
    region = img[y0:y1, x0:x1].astype(np.float64)
    if region.size < 64:
        region = img.astype(np.float64)
    gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    jxx = float(np.sum(gx * gx))
    jyy = float(np.sum(gy * gy))
    jxy = float(np.sum(gx * gy))
    denom = jxx + jyy
    if denom <= 1e-12:
        return 0.0, 0.0
    coherence = float(np.sqrt((jxx - jyy) ** 2 + (2.0 * jxy) ** 2) / denom)
    coherence = float(min(max(coherence, 0.0), 1.0))
    # Orientation of dominant gradient; ridge orientation is orthogonal but we
    # report gradient orientation consistently (a fixed, deterministic choice).
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)  # radians in (-pi/2, pi/2]
    deg = float(np.degrees(theta) % 180.0)
    return deg, coherence


def compute_image_features(resolved_path: str, *, repo_root: Path = REPO_ROOT) -> dict[str, float]:
    """Compute deterministic domain features for one image.

    Reuses ``extract_image_quality`` for canonical quality features and adds
    percentiles, entropy, local contrast, orientation coherence, and a bbox
    foreground occupancy.
    """

    quality = extract_image_quality(resolved_path, repo_root=repo_root)
    if float(quality.get("image_read_ok", 0.0)) < 1.0:
        raise DomainGapError(f"Unreadable image (quality read failed): {resolved_path}")

    img = cv2.imread(str(resolved_path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise DomainGapError(f"Unreadable image (cv2 returned None): {resolved_path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    values = img.astype(np.float64)
    p05, p25, p50, p75, p95 = (float(v) for v in np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95]))
    height, width = img.shape[:2]
    x0, y0, x1, y1 = foreground_bbox(img, margin=0)
    bbox_occ = float(((x1 - x0) * (y1 - y0)) / float(width * height)) if width and height else float("nan")
    dom_deg, coherence = structure_tensor_orientation(img)

    return {
        "width": float(quality["width"]),
        "height": float(quality["height"]),
        "aspect_ratio": float(quality["aspect_ratio"]),
        "gray_mean": float(quality["mean_intensity"]),
        "gray_std": float(quality["std_intensity"]),
        "gray_p05": p05,
        "gray_p25": p25,
        "gray_p50": p50,
        "gray_p75": p75,
        "gray_p95": p95,
        "entropy": _shannon_entropy(img),
        "contrast_proxy": float(quality["contrast_proxy"]),
        "local_contrast": _local_contrast(img),
        "foreground_ratio": float(quality["foreground_ratio"]),
        "foreground_bbox_occupancy": bbox_occ,
        "sharpness_laplacian_var": float(quality["sharpness_laplacian_var"]),
        "edge_density": float(quality["edge_density"]),
        "orientation_dominant_deg": dom_deg,
        "orientation_coherence": coherence,
    }


def build_features_table(images: pd.DataFrame, *, repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    feature_rows: list[dict[str, float]] = []
    for resolved in images["resolved_path"].astype(str):
        feature_rows.append(compute_image_features(resolved, repo_root=repo_root))
    features = pd.DataFrame(feature_rows)
    out = pd.concat([images[META_COLUMNS].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    # Deterministic output order.
    out = out.sort_values(["split", "modality", "sample_uid"], kind="mergesort").reset_index(drop=True)

    # Validate required-basic features are finite.
    for col in REQUIRED_BASIC_FEATURES:
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise DomainGapError(f"Required basic feature {col!r} has {n_bad} non-finite value(s).")
    return out


# ---------------------------------------------------------------------------
# Modality-gap summary + effect sizes
# ---------------------------------------------------------------------------
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0 if np.mean(a) == np.mean(b) else float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def build_domain_gap_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        cvals = pd.to_numeric(features.loc[features["modality"] == CONTACTLESS, feature], errors="coerce").to_numpy(float)
        tvals = pd.to_numeric(features.loc[features["modality"] == CONTACT, feature], errors="coerce").to_numpy(float)
        d = _cohens_d(cvals, tvals)  # contactless - contact
        for modality, vals in ((CONTACTLESS, cvals), (CONTACT, tvals)):
            v = vals[np.isfinite(vals)]
            rows.append(
                {
                    "feature": feature,
                    "modality": modality,
                    "count": int(len(v)),
                    "mean": float(np.mean(v)) if len(v) else float("nan"),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                    "median": float(np.median(v)) if len(v) else float("nan"),
                    "p05": float(np.quantile(v, 0.05)) if len(v) else float("nan"),
                    "p95": float(np.quantile(v, 0.95)) if len(v) else float("nan"),
                    "cohens_d_contactless_minus_contact": d,
                    "abs_cohens_d": abs(d) if np.isfinite(d) else float("nan"),
                }
            )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["feature", "modality"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paired-genuine cross-modality gaps
# ---------------------------------------------------------------------------
def build_paired_genuine_gaps(
    manifest_dir: Path, splits: list[str], features: pd.DataFrame
) -> pd.DataFrame:
    feat_by_uid = features.set_index("sample_uid")
    diff_rows: list[dict[str, Any]] = []
    for split, pairs_csv in iter_pair_split_csvs(manifest_dir, splits):
        pairs = load_polyu_cross_pairs(pairs_csv)
        genuine = pairs[pairs["label"].astype(int) == 1]
        for _, row in genuine.iterrows():
            uid_a, uid_b = str(row["sample_uid_a"]), str(row["sample_uid_b"])
            if uid_a not in feat_by_uid.index or uid_b not in feat_by_uid.index:
                continue
            fa = feat_by_uid.loc[uid_a]
            fb = feat_by_uid.loc[uid_b]
            entry: dict[str, Any] = {
                "split": str(split),
                "session_a": str(row.get("session_a", "")),
                "session_b": str(row.get("session_b", "")),
            }
            for feature in NUMERIC_FEATURES:
                entry[feature] = abs(float(fa[feature]) - float(fb[feature]))
            diff_rows.append(entry)

    if not diff_rows:
        return pd.DataFrame(columns=["split", "session_a", "session_b", "n_pairs", "feature", "mean_abs_diff", "median_abs_diff"])

    diffs = pd.DataFrame(diff_rows)
    long_rows: list[dict[str, Any]] = []
    for (split, sa, sb), group in diffs.groupby(["split", "session_a", "session_b"], sort=True):
        for feature in NUMERIC_FEATURES:
            vals = pd.to_numeric(group[feature], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            long_rows.append(
                {
                    "split": split,
                    "session_a": sa,
                    "session_b": sb,
                    "n_pairs": int(len(group)),
                    "feature": feature,
                    "mean_abs_diff": float(np.mean(vals)) if len(vals) else float("nan"),
                    "median_abs_diff": float(np.median(vals)) if len(vals) else float("nan"),
                }
            )
    out = pd.DataFrame(long_rows)
    return out.sort_values(["split", "session_a", "session_b", "feature"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    manifest_dir: Path,
    outdir: Path,
    splits: list[str],
    polyu_root: Optional[str],
) -> dict[str, Any]:
    manifest_dir = Path(manifest_dir)
    if not (manifest_dir / "manifest.csv").exists():
        raise DomainGapError(f"PolyU Cross manifest.csv not found under {manifest_dir}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)
    images = collect_bundle_images(manifest_dir, splits, resolved_root.root)
    features = build_features_table(images)

    features_csv = outdir / "image_domain_features.csv"
    features.to_csv(features_csv, index=False)
    summary = build_domain_gap_summary(features)
    summary_csv = outdir / "domain_gap_summary.csv"
    summary.to_csv(summary_csv, index=False)
    paired = build_paired_genuine_gaps(manifest_dir, splits, features)
    paired_csv = outdir / "paired_genuine_feature_gaps_by_session.csv"
    paired.to_csv(paired_csv, index=False)

    # Counts by modality x split x session (for reporting).
    counts = (
        features.groupby(["split", "modality", "session_id"]).size().reset_index(name="n_images")
        .sort_values(["split", "modality", "session_id"], kind="mergesort")
    )

    pairs_shas = {}
    for split, pairs_csv in iter_pair_split_csvs(manifest_dir, splits):
        pairs_shas[split] = {"pairs_csv": str(pairs_csv), "sha256": sha256_file(pairs_csv)}

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "manifest_dir": str(manifest_dir),
        "outdir": str(outdir),
        "splits_audited": list(splits),
        "protocol": "TRAIN and VAL only; TEST is never read (Phase 4A.1 development rule).",
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "n_images_total": int(len(features)),
        "image_counts_by_split_modality_session": counts.to_dict(orient="records"),
        "modality_counts": {str(k): int(v) for k, v in features["modality"].value_counts().to_dict().items()},
        "numeric_features": list(NUMERIC_FEATURES),
        "required_basic_features": list(REQUIRED_BASIC_FEATURES),
        "reused_utilities": {
            "quality": "src.fpbench.universal.quality.extract_image_quality",
            "foreground_bbox": "src.fpbench.deep.transforms.foreground_bbox",
            "orientation": "minimal validated structure-tensor (gradient covariance) over foreground bbox",
        },
        "omitted_features": OMITTED_FEATURES,
        "outputs": {
            "image_domain_features_csv": str(features_csv),
            "domain_gap_summary_csv": str(summary_csv),
            "paired_genuine_feature_gaps_csv": str(paired_csv),
        },
        "pairs_csv_sha256": pairs_shas,
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "constraints": {
            "trained_or_finetuned": False,
            "calibrated_scores": False,
            "selected_thresholds": False,
            "modified_manifest_or_pairs": False,
            "evaluated_test": False,
            "modified_ui_or_api": False,
            "copied_biometric_images": False,
        },
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "features": features,
        "summary": summary,
        "paired": paired,
        "counts": counts,
        "features_csv": features_csv,
        "summary_csv": summary_csv,
        "paired_csv": paired_csv,
        "manifest_json": manifest_json,
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PolyU Cross domain-gap audit (diagnostics only, TRAIN/VAL). Computes deterministic "
            "per-image domain features and modality-gap statistics. Does not train, calibrate, "
            "threshold, evaluate TEST, or modify the manifest/pairs."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--splits", type=str, default="train,val", help="Development splits only (train,val).")
    parser.add_argument("--polyu_root", type=str, default="")
    return parser


def _resolve_repo_path(raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    splits = parse_csv_list(args.splits)
    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad:
        print(f"ERROR: refusing to audit non-development splits {bad}; allowed {list(ALLOWED_SPLITS)}", file=sys.stderr)
        return 2
    try:
        result = run(
            manifest_dir=_resolve_repo_path(args.data_dir),
            outdir=_resolve_repo_path(args.outdir),
            splits=splits,
            polyu_root=str(args.polyu_root).strip() or None,
        )
    except (DomainGapError, PolyUCrossPairError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    features = result["features"]
    print("\n=== PolyU Cross domain-gap audit complete ===")
    print(f"Output dir : {result['features_csv'].parent}")
    print(f"Images     : {len(features)} (TRAIN/VAL only)")
    print("Counts by split/modality/session:")
    print(result["counts"].to_string(index=False))
    top = (
        result["summary"][["feature", "cohens_d_contactless_minus_contact", "abs_cohens_d"]]
        .drop_duplicates("feature")
        .sort_values("abs_cohens_d", ascending=False)
        .head(5)
    )
    print("\nTop 5 features by |Cohen's d| (contactless - contact):")
    print(top.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
