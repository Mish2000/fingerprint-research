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
        description="Show the read-only runtime inspection payload for the identification PostgreSQL store."
    )
    parser.add_argument("--database-url", default=None, help="Optional biometric database URL override.")
    parser.add_argument("--identity-database-url", default=None, help="Optional identity database URL override.")
    parser.add_argument(
        "--table-prefix",
        default=os.getenv("IDENTIFICATION_TABLE_PREFIX", ""),
        help="Optional identification table prefix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=args.database_url,
        identity_database_url=args.identity_database_url,
        table_prefix=args.table_prefix,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
