from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.eval_classic import balanced_limit_by_label, compute_auc_eer
from src.fpbench.matchers.minutiae_matcher import (
    MinutiaeConfig,
    match_minutiae_templates,
    extract_minutiae_template,
)

MINUTIAE_METHOD_SEMANTICS_EPOCH = "minutiae_crossing_number_aligned_v2"


def parse_file_uri(raw: str) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:"):]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _cfg_key(args: argparse.Namespace) -> tuple[Any, ...]:
    return (
        int(args.target_size),
        float(args.clahe_clip),
        (int(args.clahe_grid[0]), int(args.clahe_grid[1])),
        int(args.blur_ksize),
        not bool(args.no_preprocess),
        int(args.ridge_kernel_size),
        float(args.min_ridge_response),
        int(args.min_binary_component_area),
        int(args.min_skeleton_component_area),
        int(args.border_px),
        float(args.suppress_top_ratio),
        int(args.roi_border_px),
        float(args.min_distance),
        int(args.orientation_radius),
        int(args.max_minutiae),
        int(args.max_thinning_iterations),
        float(args.candidate_quality_floor),
        int(args.dense_bifurcation_radius),
        float(args.max_dense_bifurcation_density),
        float(args.bifurcation_suppression_scale),
        float(args.spatial_tolerance),
        float(args.angle_tolerance_deg),
        int(args.anchor_limit),
        int(args.min_required_minutiae),
    )


def _config_from_key(key: tuple[Any, ...]) -> MinutiaeConfig:
    return MinutiaeConfig(
        target_size=int(key[0]),
        clahe_clip=float(key[1]),
        clahe_grid=tuple(key[2]),
        blur_ksize=int(key[3]),
        preprocess=bool(key[4]),
        ridge_kernel_size=int(key[5]),
        min_ridge_response=float(key[6]),
        min_binary_component_area=int(key[7]),
        min_skeleton_component_area=int(key[8]),
        border_px=int(key[9]),
        suppress_top_ratio=float(key[10]),
        roi_border_px=int(key[11]),
        min_distance=float(key[12]),
        orientation_radius=int(key[13]),
        max_minutiae=int(key[14]),
        max_thinning_iterations=int(key[15]),
        candidate_quality_floor=float(key[16]),
        dense_bifurcation_radius=int(key[17]),
        max_dense_bifurcation_density=float(key[18]),
        bifurcation_suppression_scale=float(key[19]),
        spatial_tolerance=float(key[20]),
        angle_tolerance_deg=float(key[21]),
        anchor_limit=int(key[22]),
        min_required_minutiae=int(key[23]),
    )


def _config_hash(key: tuple[Any, ...]) -> str:
    payload = json.dumps(key, sort_keys=True, default=list, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TemplateCache:
    def __init__(self, cfg: MinutiaeConfig, config_hash: str, *, enabled: bool = True):
        self.cfg = cfg
        self.config_hash = str(config_hash)
        self.enabled = bool(enabled)
        self._cache: dict[tuple[str, str], Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, path_str: str):
        path = parse_file_uri(path_str)
        cache_key = (str(path), self.config_hash)
        if self.enabled and cache_key in self._cache:
            self.hits += 1
            return self._cache[cache_key]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        template = extract_minutiae_template(img, mask=None, cfg=self.cfg)
        self.misses += 1
        if self.enabled:
            self._cache[cache_key] = template
        return template


def _meta_payload(
    args: argparse.Namespace,
    *,
    auc: float,
    eer: float,
    avg_ms_pair: float,
    n_pairs: int,
    config_hash: str,
    cache: TemplateCache,
) -> dict[str, Any]:
    return {
        "schema_version": "v1_minutiae_scores_meta",
        "method": "minutiae",
        "method_semantics_epoch": MINUTIAE_METHOD_SEMANTICS_EPOCH,
        "split": args.split,
        "n_pairs": int(n_pairs),
        "auc": float(auc),
        "eer": float(eer),
        "avg_ms_pair": float(avg_ms_pair),
        "template_cache": {
            "enabled": bool(cache.enabled),
            "config_hash": str(config_hash),
            "hits": int(cache.hits),
            "misses": int(cache.misses),
            "entries": int(len(cache._cache)),
            "key": "resolved_image_path_plus_config_hash",
        },
        "config": {
            "preprocess": "runtime_square_512_clahe_blur" if not args.no_preprocess else "disabled",
            "extraction": "polarity_validated_ridge_enhancement_otsu_skeleton_crossing_number",
            "skeletonization": "zhang_suen_numpy",
            "crossing_number": {"ridge_ending": 1, "bifurcation": 3},
            "alignment": "anchor_orientation_rotation_translation",
            "score_mode": "quality_penalized_matched_minutiae_over_min_template_count_floor",
            "method_semantics_epoch": MINUTIAE_METHOD_SEMANTICS_EPOCH,
            "target_size": int(args.target_size),
            "max_minutiae": int(args.max_minutiae),
            "candidate_quality_floor": float(args.candidate_quality_floor),
            "dense_bifurcation_radius": int(args.dense_bifurcation_radius),
            "max_dense_bifurcation_density": float(args.max_dense_bifurcation_density),
            "bifurcation_suppression_scale": float(args.bifurcation_suppression_scale),
            "spatial_tolerance": float(args.spatial_tolerance),
            "angle_tolerance_deg": float(args.angle_tolerance_deg),
            "min_required_minutiae": int(args.min_required_minutiae),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate classical crossing-number minutiae matching.")
    ap.add_argument("out_csv", type=str)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--blur_ksize", type=int, default=3)
    ap.add_argument("--no_preprocess", action="store_true")
    ap.add_argument("--ridge_kernel_size", type=int, default=15)
    ap.add_argument("--min_ridge_response", type=float, default=5.0)
    ap.add_argument("--min_binary_component_area", type=int, default=12)
    ap.add_argument("--min_skeleton_component_area", type=int, default=8)
    ap.add_argument("--border_px", type=int, default=16)
    ap.add_argument("--suppress_top_ratio", type=float, default=0.03)
    ap.add_argument("--roi_border_px", type=int, default=10)
    ap.add_argument("--min_distance", type=float, default=8.0)
    ap.add_argument("--orientation_radius", type=int, default=9)
    ap.add_argument("--max_minutiae", type=int, default=96)
    ap.add_argument("--max_thinning_iterations", type=int, default=80)
    ap.add_argument("--candidate_quality_floor", type=float, default=0.20)
    ap.add_argument("--dense_bifurcation_radius", type=int, default=5)
    ap.add_argument("--max_dense_bifurcation_density", type=float, default=0.24)
    ap.add_argument("--bifurcation_suppression_scale", type=float, default=1.35)
    ap.add_argument("--spatial_tolerance", type=float, default=14.0)
    ap.add_argument("--angle_tolerance_deg", type=float, default=30.0)
    ap.add_argument("--anchor_limit", type=int, default=28)
    ap.add_argument("--min_required_minutiae", type=int, default=12)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--disable_template_cache", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for a future deterministic Windows-safe worker pool; values >1 currently fall back to serial caching.",
    )
    args = ap.parse_args()

    out_path = parse_file_uri(args.out_csv)
    pairs_path = parse_file_uri(args.pairs) if args.pairs else (
        REPO_ROOT / "data" / "processed" / "nist_sd300b" / f"pairs_{args.split}.csv"
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = pd.read_csv(pairs_path)
    if args.limit and args.limit > 0:
        pairs = balanced_limit_by_label(pairs, label_col="label", limit=int(args.limit))

    key = _cfg_key(args)
    cfg = _config_from_key(key)
    config_hash = _config_hash(key)
    if int(args.workers) != 1:
        print(
            "[minutiae] --workers > 1 requested, but multiprocessing is intentionally disabled "
            "for this pass to keep Windows execution deterministic with the in-memory template cache."
        )
    template_cache = TemplateCache(cfg, config_hash, enabled=not bool(args.disable_template_cache))
    rows = []
    t0 = time.perf_counter()
    total_pairs = int(len(pairs))
    progress_every = max(0, int(args.progress_every))
    for index, (_, row) in enumerate(pairs.iterrows(), start=1):
        path_a = str(row["path_a"])
        path_b = str(row["path_b"])
        label = int(row["label"])
        row_split = str(row.get("split", args.split))
        template_a = template_cache.get(path_a)
        template_b = template_cache.get(path_b)
        result = match_minutiae_templates(template_a, template_b, cfg=cfg)
        rows.append(
            {
                "label": label,
                "split": row_split,
                "path_a": path_a,
                "path_b": path_b,
                "score": float(result.score),
                "raw_alignment_score": float(result.raw_alignment_score),
                "score_multiplier": float(result.score_multiplier),
                "score_component_template_quality": float(
                    result.score_components.get("template_quality_multiplier", 0.0)
                ),
                "score_component_ambiguity": float(result.score_components.get("ambiguity_multiplier", 0.0)),
                "score_component_transform_plausibility": float(
                    result.score_components.get("transform_plausibility_multiplier", 0.0)
                ),
                "matched_minutiae": int(result.matched_count),
                "tentative_minutiae": int(result.tentative_count),
                "minutiae_count_a": int(result.minutiae_count_a),
                "minutiae_count_b": int(result.minutiae_count_b),
                "total_kept_minutiae_a": int(result.minutiae_count_a),
                "total_kept_minutiae_b": int(result.minutiae_count_b),
                "minutiae_a": int(result.minutiae_count_a),
                "minutiae_b": int(result.minutiae_count_b),
                "endings_a": int(result.endings_a),
                "endings_b": int(result.endings_b),
                "bifurcations_a": int(result.bifurcations_a),
                "bifurcations_b": int(result.bifurcations_b),
                "kept_endings_a": int(result.endings_a),
                "kept_endings_b": int(result.endings_b),
                "kept_bifurcations_a": int(result.bifurcations_a),
                "kept_bifurcations_b": int(result.bifurcations_b),
                "skeleton_foreground_pixels_a": int(result.skeleton_foreground_pixels_a),
                "skeleton_foreground_pixels_b": int(result.skeleton_foreground_pixels_b),
                "skeleton_density_a": float(result.skeleton_density_a),
                "skeleton_density_b": float(result.skeleton_density_b),
                "raw_candidate_endings_a": int(result.raw_candidate_endings_a),
                "raw_candidate_endings_b": int(result.raw_candidate_endings_b),
                "raw_candidate_bifurcations_a": int(result.raw_candidate_bifurcations_a),
                "raw_candidate_bifurcations_b": int(result.raw_candidate_bifurcations_b),
                "saturated_by_max_minutiae_a": bool(result.saturated_by_max_minutiae_a),
                "saturated_by_max_minutiae_b": bool(result.saturated_by_max_minutiae_b),
                "ridge_polarity_a": result.ridge_polarity_a,
                "ridge_polarity_b": result.ridge_polarity_b,
                "polarity_used_a": result.ridge_polarity_a,
                "polarity_used_b": result.ridge_polarity_b,
                "extraction_quality_flags_a": ";".join(result.extraction_quality_flags_a),
                "extraction_quality_flags_b": ";".join(result.extraction_quality_flags_b),
                "transform_angle_deg": result.transform_angle_deg,
                "transform_dx": result.transform_dx,
                "transform_dy": result.transform_dy,
            }
        )
        if progress_every and (index % progress_every == 0 or index == total_pairs):
            elapsed_s = time.perf_counter() - t0
            print(
                f"[minutiae] {index}/{total_pairs} pairs | "
                f"cache hits={template_cache.hits} misses={template_cache.misses} | "
                f"elapsed={elapsed_s:.1f}s"
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    df = pd.DataFrame(rows)
    auc, eer = compute_auc_eer(df["label"].values, df["score"].values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            _meta_payload(
                args,
                auc=auc,
                eer=eer,
                avg_ms_pair=elapsed_ms / max(len(df), 1),
                n_pairs=len(df),
                config_hash=config_hash,
                cache=template_cache,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Split={args.split} | N={len(df)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print("Saved:", out_path)
    print("Meta :", meta_path)


if __name__ == "__main__":
    main()
