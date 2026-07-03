from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.generate_plain_roll_train_scores_v2 import (
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    DEFAULT_OUTDIR,
    TrainScoreGenerationError,
    parse_file_uri,
    validate_existing_artifacts,
)


def _parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit SD300 anatomical train score tables against canonical train pair bundles."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = parse_file_uri(args.repo_root)
    outdir = parse_file_uri(args.outdir, repo_root=repo_root)
    datasets = _parse_csv_arg(args.datasets)
    methods = _parse_csv_arg(args.methods)
    try:
        rows = validate_existing_artifacts(outdir=outdir, datasets=datasets, methods=methods, repo_root=repo_root)
    except TrainScoreGenerationError as exc:
        print(f"Train score audit failed: {exc}", file=sys.stderr)
        return 2

    summary_path = outdir / "score_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print("Train score audit passed.")
    print(f"summary: {summary_path}")
    for row in rows:
        print(
            "{dataset} {method}: rows={rows} pos={pos} neg={neg} frgp={frgp} sha={sha}".format(
                dataset=row["dataset"],
                method=row["method"],
                rows=row["rows"],
                pos=row["positive_count"],
                neg=row["negative_count"],
                frgp=row["frgp_coverage"],
                sha=row["pair_source_sha256"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
