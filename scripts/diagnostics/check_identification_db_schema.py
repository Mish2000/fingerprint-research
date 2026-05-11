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
        description="Read-only schema and runtime diagnostics for the PostgreSQL-backed identification store."
    )
    parser.add_argument("--database-url", default=None, help="Optional biometric database URL override.")
    parser.add_argument("--identity-database-url", default=None, help="Optional identity database URL override.")
    parser.add_argument(
        "--table-prefix",
        default=os.getenv("IDENTIFICATION_TABLE_PREFIX", ""),
        help="Optional identification table prefix.",
    )
    return parser.parse_args(argv)


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _count_text(value: object) -> str:
    return "unknown" if value is None else str(value)


def _format_vectors(counts: dict[str, object]) -> str:
    ordered = []
    for method in sorted(counts):
        ordered.append(f"{method}={_count_text(counts[method])}")
    return ", ".join(ordered) or "(none)"


def _format_generic_vectors(counts: dict[str, object]) -> str:
    ordered = []
    for key in sorted(counts):
        ordered.append(f"{key}={_count_text(counts[key])}")
    return ", ".join(ordered) or "(none)"


def _print_payload(payload: dict[str, object]) -> None:
    urls = dict(payload["redacted_database_urls"])
    tables = dict(payload["resolved_table_names"])
    presence = dict(payload["table_presence"])
    counts = dict(payload["row_counts"])
    issues = list(payload["issues"])
    readiness = dict(payload.get("readiness", {}))
    hardening = dict(payload.get("schema_hardening", {}))
    template_protection = dict(payload.get("template_protection", {}))

    print("[layout]")
    print(f"  backend: {payload['backend']}")
    print(f"  layout_version: {payload['layout_version']}")
    print(f"  dual_database_enabled: {_bool_text(bool(payload['dual_database_enabled']))}")
    print(f"  table_prefix: {payload['table_prefix'] or '(none)'}")

    print("\n[database_urls]")
    print(f"  biometric_db: {urls['biometric_db']}")
    print(f"  identity_db:  {urls['identity_db']}")

    print("\n[resolved_tables]")
    print(f"  person:   {tables['person']}")
    print(f"  identity: {tables['identity']}")
    print(f"  raw:      {tables['raw']}")
    print(f"  vectors:  {tables['vectors']}")
    if "generic_vectors" in tables:
        print(f"  generic_vectors: {tables['generic_vectors']}")

    print("\n[table_presence]")
    for database_role in ("biometric_db", "identity_db"):
        role_presence = dict(presence[database_role])
        fields = ", ".join(
            f"{name}={_bool_text(bool(role_presence.get(name)))}"
            for name in ("person", "identity", "raw", "vectors", "generic_vectors")
        )
        print(f"  {database_role}: {fields}")

    print("\n[row_counts]")
    print(f"  people: { _count_text(counts['people']) }")
    print(f"  identity: { _count_text(counts['identity']) }")
    print(f"  raw: { _count_text(counts['raw']) }")
    print(f"  vectors_by_method: {_format_vectors(dict(counts['vectors_by_method']))}")
    print(
        "  generic_vectors_by_method_kind: "
        + _format_generic_vectors(dict(counts.get("generic_vectors_by_method_kind", {})))
    )
    print(
        "  vector_extension_present_in_biometric_db: "
        + _bool_text(payload["vector_extension_present_in_biometric_db"])
    )

    vector_storage = dict(payload.get("vector_storage_schema", {}))
    if vector_storage:
        print("\n[vector_storage]")
        print(f"  mode: {vector_storage.get('mode', 'unknown')}")
        print(
            "  method_generic_vectors_supported: "
            + _bool_text(vector_storage.get("method_generic_vectors_supported"))
        )
        print(
            "  generic_retrieval_vectors_table_present: "
            + _bool_text(vector_storage.get("generic_retrieval_vectors_table_present"))
        )
        print(
            "  generic_schema_compatible: "
            + _bool_text(vector_storage.get("generic_schema_compatible"))
        )
        print("  legacy_fallback_used: " + _bool_text(vector_storage.get("legacy_fallback_used")))
        print("  backfill_recommended: " + _bool_text(vector_storage.get("backfill_recommended")))

    if hardening:
        guarantees = dict(hardening.get("identity_map_guarantees", {}))
        drift = dict(hardening.get("drift", {}))
        print("\n[schema_hardening]")
        print(f"  contract_enforced: {_bool_text(guarantees.get('contract_enforced'))}")
        print(f"  completeness_guaranteed: {_bool_text(guarantees.get('completeness_guaranteed'))}")
        print(
            "  missing_constraints: "
            + (", ".join(drift.get("missing_constraints", [])) or "(none)")
        )
        print(
            "  missing_indexes: "
            + (", ".join(drift.get("missing_indexes", [])) or "(none)")
        )

    if template_protection:
        print("\n[template_protection]")
        print(f"  raw_image_storage_policy: {template_protection.get('raw_image_storage_policy', 'unknown')}")
        print(
            "  new_raw_image_persistence_enabled: "
            + _bool_text(template_protection.get("new_raw_image_persistence_enabled"))
        )
        print(
            "  raw_image_bytes_column_present: "
            + _bool_text(template_protection.get("raw_image_bytes_column_present"))
        )
        print(
            "  raw_image_bytes_column_nullable: "
            + _bool_text(template_protection.get("raw_image_bytes_column_nullable"))
        )
        print(
            "  legacy_raw_image_storage_status: "
            + str(template_protection.get("legacy_raw_image_storage_status", "unknown"))
        )
        print(
            "  legacy_raw_image_bytes_row_count: "
            + _count_text(template_protection.get("legacy_raw_image_bytes_row_count"))
        )
        sample = [str(item) for item in list(template_protection.get("legacy_raw_image_bytes_sample", []))]
        print("  legacy_raw_image_bytes_sample: " + (", ".join(sample) or "(none)"))

    print("\n[readiness]")
    print(f"  status: {readiness.get('status', 'unknown')}")
    print(f"  ready: {_bool_text(readiness.get('ready'))}")
    print(f"  errors: {readiness.get('error_count', len(payload.get('errors', [])))}")
    print(f"  warnings: {readiness.get('warning_count', len(payload.get('warnings', [])))}")

    print("\n[issues]")
    if not issues:
        print("  (none)")
    else:
        for issue in issues:
            severity = str(issue["severity"]).upper()
            print(f"  [{severity}] {issue['code']}: {issue['message']}")

    print("\nsummary:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=args.database_url,
        identity_database_url=args.identity_database_url,
        table_prefix=args.table_prefix,
    )
    _print_payload(payload)
    return 0 if bool(payload["overall_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
