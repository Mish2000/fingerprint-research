from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BATCH_TARGET = "pipelines.benchmark.run_benchmark_matrix"
ONCE_TARGET = "scripts.run_benchmark_once"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility shim for legacy benchmark runner.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval_all", action="store_true", help="Compatibility flag for batch benchmark runs.")
    mode.add_argument("--eval_one", action="store_true", help="Compatibility flag for a single benchmark run.")

    parser.add_argument("--dataset", type=str, default="nist_sd300b")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--outdir", type=str, default="artifacts/reports/benchmark/current")
    parser.add_argument("--method", type=str, default="classic_v2")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--methods", type=str, default="classic_v2,harris,sift,dl_quick,dedicated,vit")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ensure_pairs", action="store_true")
    parser.add_argument("--dedicated_ckpt", type=str, default="auto")
    parser.add_argument("--emb_cache_dir", type=str, default="")
    parser.add_argument("--cache_write", action="store_true")
    parser.add_argument("--cache_strip_prefix", type=str, default="")
    parser.add_argument("--fusion_fit_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--fusion_sift_weight", type=float, default=0.91)
    parser.add_argument("--fusion_dl_weight", type=float, default=0.05)
    parser.add_argument("--fusion_vit_weight", type=float, default=0.04)
    return parser


def translate_legacy_args(args: argparse.Namespace) -> Tuple[str, list[str]]:
    target = ONCE_TARGET if args.eval_one else BATCH_TARGET
    translated = [
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--outdir", args.outdir,
        "--limit", str(args.limit),
        "--dedicated_ckpt", args.dedicated_ckpt,
        "--emb_cache_dir", args.emb_cache_dir,
        "--cache_strip_prefix", args.cache_strip_prefix,
    ]

    if args.ensure_pairs:
        translated.append("--ensure_pairs")
    if args.cache_write:
        translated.append("--cache_write")

    if target == BATCH_TARGET:
        translated += [
            "--methods", args.methods,
            "--splits", args.splits,
            "--fusion_fit_split", args.fusion_fit_split,
            "--fusion_sift_weight", str(args.fusion_sift_weight),
            "--fusion_dl_weight", str(args.fusion_dl_weight),
            "--fusion_vit_weight", str(args.fusion_vit_weight),
        ]
    else:
        translated += [
            "--method", args.method,
            "--split", args.split,
        ]
        if args.method == "fusion_balanced_v1":
            translated += [
                "--fusion_fit_split", args.fusion_fit_split,
                "--fusion_sift_weight", str(args.fusion_sift_weight),
                "--fusion_dl_weight", str(args.fusion_dl_weight),
                "--fusion_vit_weight", str(args.fusion_vit_weight),
            ]

    return target, translated


def dispatch_legacy_args(target: str, argv: list[str]) -> int:
    module = importlib.import_module(target)
    return int(module.main(argv))


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(
        "WARNING: scripts/run_all.py is a compatibility shim. "
        "Canonical entry points: pipelines/benchmark/run_benchmark_matrix.py "
        "and scripts/run_benchmark_once.py.",
        file=sys.stderr,
    )

    target, translated = translate_legacy_args(args)
    return dispatch_legacy_args(target, translated)


if __name__ == "__main__":
    raise SystemExit(main())
