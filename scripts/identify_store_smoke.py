from __future__ import annotations

import argparse
import os

from apps.api.identification_service import IdentificationService
from apps.api.schemas import MatchMethod
from src.fpbench.identification.secure_split_store import (
    IDENTIFICATION_RETRIEVAL_VECTOR_METHODS,
    IdentifyHints,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Small smoke helper for the PostgreSQL-backed 1:N identification store.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--identity-database-url", default=os.getenv("IDENTITY_DATABASE_URL"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--image", required=True)
    p_enroll.add_argument("--full-name", required=True)
    p_enroll.add_argument("--national-id", required=True)
    p_enroll.add_argument("--capture", default="plain")
    p_enroll.add_argument(
        "--vector-methods",
        default="dl,vit",
        help=(
            "Comma-separated vector-backed enrollment methods. "
            f"Supported shortlist retrieval methods: {sorted(IDENTIFICATION_RETRIEVAL_VECTOR_METHODS)}"
        ),
    )
    p_enroll.add_argument("--replace-existing", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("--image", required=True)
    p_search.add_argument("--capture", default="plain")
    p_search.add_argument(
        "--retrieval-method",
        default="dl",
        choices=sorted(IDENTIFICATION_RETRIEVAL_VECTOR_METHODS),
        help="Shortlist retrieval method backed by stored pgvector columns.",
    )
    p_search.add_argument("--rerank-method", default="sift", choices=[m.value for m in MatchMethod])
    p_search.add_argument("--shortlist-size", type=int, default=25)
    p_search.add_argument("--name-pattern")
    p_search.add_argument("--national-id-pattern")
    p_search.add_argument("--created-from")
    p_search.add_argument("--created-to")

    args = parser.parse_args()
    svc = IdentificationService(
        database_url=args.database_url,
        identity_database_url=args.identity_database_url,
    )

    if args.cmd == "enroll":
        receipt = svc.enroll_from_path(
            path=args.image,
            full_name=args.full_name,
            national_id=args.national_id,
            capture=args.capture,
            vector_methods=[item.strip() for item in args.vector_methods.split(",") if item.strip()],
            replace_existing=bool(args.replace_existing),
        )
        print(
            {
                "random_id": receipt.random_id,
                "created_at": receipt.created_at,
                "vector_methods": receipt.vector_methods,
                "storage_layout": svc.store.dump_layout(),
            }
        )
        return 0

    result = svc.identify_from_path(
        path=args.image,
        capture=args.capture,
        retrieval_method=args.retrieval_method,
        rerank_method=MatchMethod(args.rerank_method),
        shortlist_size=args.shortlist_size,
        hints=IdentifyHints(
            name_pattern=args.name_pattern,
            national_id_pattern=args.national_id_pattern,
            created_from=args.created_from,
            created_to=args.created_to,
        ),
    )
    print(
        {
            "decision": result.decision,
            "top_candidate": result.top_candidate.__dict__ if result.top_candidate else None,
            "candidate_pool_size": result.candidate_pool_size,
            "shortlist_size": result.shortlist_size,
            "latency_ms": result.latency_ms,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

