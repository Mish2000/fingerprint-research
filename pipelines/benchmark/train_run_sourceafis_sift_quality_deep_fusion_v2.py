from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT_CANDIDATE = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))

from src.fpbench.universal.deep_fusion_v2 import (
    DEFAULT_DATASETS,
    DEFAULT_SPLITS,
    DEFAULT_TARGET_FARS,
    GROUP_WEIGHTED_METHOD,
    METHOD_NAME,
    VARIANTS,
    parse_group_weights,
    parse_csv_list,
    parse_float_list,
    run_variants,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the official SourceAFIS+SIFT+Quality+Deep Fusion v2 model. "
            "The model is fit on TRAIN only, thresholds are selected from VAL negatives only, "
            "and TEST is evaluated with frozen thresholds."
        )
    )
    parser.add_argument("--repo-root", required=True, help="Repository root, e.g. C:\\fingerprint-research")
    parser.add_argument(
        "--outdir",
        default="artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep",
        help="Output directory. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--target-fars", default=",".join(str(x) for x in DEFAULT_TARGET_FARS))
    parser.add_argument(
        "--variants",
        default=METHOD_NAME,
        help=(
            "Comma-separated variant names. Default runs only the official v2. "
            "Use --run-ablation for the recommended full ablation suite."
        ),
    )
    parser.add_argument("--run-ablation", action="store_true", help="Run the full recommended ablation suite.")
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Disable quality extraction. Intended only for fast smoke/debug runs; not the official v2 result.",
    )
    parser.add_argument("--save-training-table", action="store_true")
    parser.add_argument(
        "--run-group-weighted",
        action="store_true",
        help=(
            "Run the group-weighted Fusion v2 experiment as the main output method. "
            "This uses SourceAFIS/SIFT/deep/quality group weights after normalization and before logistic regression."
        ),
    )
    parser.add_argument(
        "--group-weights",
        default=None,
        help=(
            "Manual group weights, e.g. sourceafis=45,sift=15,deep=30,quality=10. "
            "Values are normalized to sum to 1.0 before being applied."
        ),
    )
    parser.add_argument(
        "--auto-group-weights",
        action="store_true",
        help="Estimate group weights automatically from VAL performance only; TEST is not used.",
    )
    parser.add_argument(
        "--auto-weight-metric",
        default="auc",
        choices=("auc", "tar_at_far", "eer_complement"),
        help="VAL metric used when --auto-group-weights is enabled.",
    )
    parser.add_argument(
        "--auto-weight-target-far",
        type=float,
        default=0.01,
        help="Target FAR used only when --auto-weight-metric=tar_at_far.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir

    if args.run_ablation:
        variants = [
            "sourceafis_only_calibrated",
            "sourceafis_sift_score",
            "sourceafis_sift_geometry",
            "sourceafis_sift_quality",
            "sourceafis_sift_deep_score",
            "sourceafis_sift_deep_logit",
            METHOD_NAME,
        ]
        if args.run_group_weighted:
            variants.append(GROUP_WEIGHTED_METHOD)
    elif args.run_group_weighted:
        variants = [GROUP_WEIGHTED_METHOD]
    else:
        variants = list(parse_csv_list(args.variants))

    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {sorted(VARIANTS)}")

    results = run_variants(
        repo_root=repo_root,
        outdir=outdir,
        datasets=parse_csv_list(args.datasets),
        splits=parse_csv_list(args.splits),
        target_fars=parse_float_list(args.target_fars),
        variants=variants,
        include_quality_override=False if args.no_quality else None,
        save_training_table=bool(args.save_training_table),
        group_weights=parse_group_weights(args.group_weights) if args.group_weights else None,
        auto_group_weights=bool(args.auto_group_weights),
        group_weight_metric=str(args.auto_weight_metric),
        group_weight_target_far=float(args.auto_weight_target_far),
    )

    print("[done]")
    print(results["metrics"])
    print("metrics:", outdir / "plain_roll_final_metrics.csv")
    print("thresholds:", outdir / "plain_roll_final_thresholds.csv")
    print("summary:", outdir / "plain_roll_final_summary.md")
    print("ablation_metrics:", outdir / "ablation_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
