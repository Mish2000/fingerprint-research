from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.identification.secure_split_store import SecureSplitFingerprintStore


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally reconcile the PostgreSQL-backed identification runtime store."
    )
    parser.add_argument("--database-url", default=None, help="Optional biometric database URL override.")
    parser.add_argument("--identity-database-url", default=None, help="Optional identity database URL override.")
    parser.add_argument(
        "--table-prefix",
        default=os.getenv("IDENTIFICATION_TABLE_PREFIX", ""),
        help="Optional identification table prefix.",
    )
    parser.add_argument(
        "--repair-raw-orphans",
        action="store_true",
        help="Delete raw_fingerprints rows that have no matching person_directory row.",
    )
    parser.add_argument(
        "--repair-vector-orphans",
        action="store_true",
        help="Delete feature_vectors rows that have no matching person_directory row.",
    )
    parser.add_argument(
        "--repair-identity-orphans",
        action="store_true",
        help="Delete identity_map rows that have no matching person_directory row.",
    )
    parser.add_argument(
        "--redact-legacy-raw-image-bytes",
        "--redact-legacy-image-bytes",
        dest="redact_legacy_raw_image_bytes",
        action="store_true",
        help="Set legacy raw_fingerprints.image_bytes payloads to NULL without reading or backing up bytes.",
    )
    parser.add_argument(
        "--backfill-generic-retrieval-vectors",
        action="store_true",
        help="Copy legacy feature_vectors rows into method_retrieval_vectors without re-embedding.",
    )
    return parser.parse_args(argv)


def _repair_actions_from_args(args: argparse.Namespace) -> list[str]:
    actions: list[str] = []
    if args.repair_raw_orphans:
        actions.append("repair_raw_orphans")
    if args.repair_vector_orphans:
        actions.append("repair_vector_orphans")
    if args.repair_identity_orphans:
        actions.append("repair_identity_orphans")
    if args.redact_legacy_raw_image_bytes:
        actions.append("redact_legacy_raw_image_bytes")
    if args.backfill_generic_retrieval_vectors:
        actions.append("backfill_generic_retrieval_vectors")
    return actions


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = SecureSplitFingerprintStore.reconcile_runtime_state(
        database_url=args.database_url,
        identity_database_url=args.identity_database_url,
        table_prefix=args.table_prefix,
        repair_actions=_repair_actions_from_args(args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    summary = dict(report.get("summary", {}))
    return 0 if bool(summary.get("overall_ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
