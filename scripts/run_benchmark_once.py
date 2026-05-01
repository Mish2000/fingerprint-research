from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.benchmark import run_benchmark_matrix as matrix


SUPPORTED_METHODS = [
    "classic_v2",
    "harris",
    "sift",
    "dl_quick",
    "dedicated",
    "vit",
    matrix.FUSION_METHOD,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical single-run benchmark runner.")
    parser.add_argument("--dataset", type=str, default="nist_sd300b")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--outdir", type=str, default="artifacts/reports/benchmark/current")
    parser.add_argument("--method", type=str, default="classic_v2", choices=SUPPORTED_METHODS)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
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


def run_once(args: argparse.Namespace) -> int:
    outdir = matrix.resolve_path(args.outdir)
    matrix.ensure_dir(outdir)

    log_path = outdir / "run.log"
    if log_path.exists():
        log_path.unlink()

    def log(message: str) -> None:
        print(message)
        matrix.write_log_line(log_path, message)

    try:
        data_dir = matrix.resolve_dataset_dir(args.dataset, args.data_dir)
    except Exception as exc:
        log(f"ERROR: Could not resolve dataset dir: {exc}")
        return 2

    required_files = matrix.required_files_for_data_dir(data_dir)
    missing_required = [path for path in required_files if not path.exists()]
    if missing_required:
        log("ERROR: Missing required dataset metadata files:")
        for path in missing_required:
            log(f"  - {path}")
        return 2

    optional_files = [path for path in matrix.optional_files_for_data_dir(data_dir) if path.exists()]
    input_hashes = {str(path): matrix.sha256_file(path) for path in required_files + optional_files}

    emb_cache_dir = str(matrix.resolve_path(args.emb_cache_dir)) if args.emb_cache_dir else ""
    dedicated_ckpt = (
        matrix.resolve_dedicated_ckpt(args.dedicated_ckpt)
        if args.method == "dedicated"
        else args.dedicated_ckpt
    )

    manifest = matrix.build_manifest_payload(
        dataset=args.dataset,
        data_dir=data_dir,
        outdir=outdir,
        methods=[args.method],
        splits=[args.split],
        limit=args.limit,
        ensure_pairs=args.ensure_pairs,
        emb_cache_dir=emb_cache_dir,
        cache_write=args.cache_write,
        cache_strip_prefix=args.cache_strip_prefix,
        dedicated_ckpt=dedicated_ckpt,
        fusion_fit_split=args.fusion_fit_split,
        fusion_sift_weight=args.fusion_sift_weight,
        fusion_dl_weight=args.fusion_dl_weight,
        fusion_vit_weight=args.fusion_vit_weight,
        input_hashes=input_hashes,
        mode="single",
        argv=list(sys.argv),
    )
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_csv = outdir / "results_summary.csv"
    summary_md = outdir / "results_summary.md"
    if summary_csv.exists():
        summary_csv.unlink()
    if summary_md.exists():
        summary_md.unlink()

    log("=== Canonical single benchmark run ===")
    log(f"Repo root : {matrix.ROOT}")
    log(f"Dataset   : {args.dataset}")
    log(f"Data dir  : {data_dir}")
    log(f"Outdir    : {outdir}")
    log(f"Method    : {args.method}")
    log(f"Split     : {args.split}")
    if emb_cache_dir:
        log(f"Emb cache : {emb_cache_dir}")

    if not matrix.EVALUATE_PY.exists():
        log(f"ERROR: Missing evaluate.py -> {matrix.EVALUATE_PY}")
        return 3

    if args.method == "dedicated":
        log(f"[DEDICATED] using ckpt: {dedicated_ckpt}")

    cmd = matrix.build_eval_cmd(
        outdir=outdir,
        dataset=args.dataset,
        data_dir=data_dir,
        method=args.method,
        split=args.split,
        limit=args.limit,
        ensure_pairs=args.ensure_pairs,
        dedicated_ckpt=dedicated_ckpt,
        emb_cache_dir=emb_cache_dir,
        cache_write=args.cache_write,
        cache_strip_prefix=args.cache_strip_prefix,
        fusion_fit_split=args.fusion_fit_split,
        fusion_sift_weight=args.fusion_sift_weight,
        fusion_dl_weight=args.fusion_dl_weight,
        fusion_vit_weight=args.fusion_vit_weight,
    )

    log("[RUN] " + " ".join(cmd))
    rc = matrix.run_cmd_stream(cmd, cwd=matrix.ROOT, log_path=log_path)
    log(f"[DONE] method={args.method} split={args.split} exit_code={rc}")
    if rc != 0:
        log("ERROR: Stopped due to non-zero exit code.")
        return 4

    matrix.render_results_md(summary_csv, summary_md)
    log(f"OK: Wrote {summary_md}")

    expected_outputs = matrix.expected_output_paths(outdir, [args.method], [args.split])
    missing_outputs = [path for path in expected_outputs if not path.exists()]
    if missing_outputs:
        log("ERROR: Missing expected outputs:")
        for path in missing_outputs:
            log(f"  - {path}")
        return 5

    log("Done.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
