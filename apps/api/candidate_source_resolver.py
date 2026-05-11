from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from apps.api.catalog_store import (
    CatalogApiError,
    load_catalog_identify_demo_identity_records,
    load_catalog_identify_seed_records,
)

SOURCE_DEMO_CATALOG = "demo_catalog_source"
SOURCE_BROWSER_CATALOG = "browser_catalog_source"
SOURCE_TEST_FIXTURE = "test_fixture_source"
SOURCE_NO_CANDIDATE_SOURCE = "no_candidate_source"
SOURCE_CANDIDATE_SOURCE_MISSING = "candidate_source_missing"

_ID_PREFIX_RE = re.compile(r"[^a-z0-9_]+")
_AVAILABLE_SOURCE_STATUSES = {
    SOURCE_DEMO_CATALOG,
    SOURCE_BROWSER_CATALOG,
    SOURCE_TEST_FIXTURE,
}


def _normalize_source_id(value: str) -> str:
    return _ID_PREFIX_RE.sub("_", str(value).strip().lower()).strip("_")


def _demo_random_id(dataset: str, identity_id: str) -> str:
    return f"demo_identify_{_normalize_source_id(dataset)}_{_normalize_source_id(identity_id)}"


def _browser_random_id(dataset: str, identity_id: str) -> str:
    return f"browser_identify_{_normalize_source_id(dataset)}_{_normalize_source_id(identity_id)}"


@dataclass(frozen=True)
class CandidateSourceResolution:
    status: str
    path: Optional[Path] = None
    capture: Optional[str] = None
    source_kind: Optional[str] = None
    detail: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.path is not None and self.status in _AVAILABLE_SOURCE_STATUSES


class CandidateSourceResolver:
    """Resolve approved non-DB candidate image sources for transient reranking."""

    def __init__(
        self,
        *,
        test_sources: Mapping[str, str | Path] | None = None,
        test_captures: Mapping[str, str] | None = None,
    ) -> None:
        self._test_sources = {str(key): Path(value) for key, value in (test_sources or {}).items()}
        self._test_captures = {str(key): str(value) for key, value in (test_captures or {}).items()}

    def register_test_source(self, random_id: str, path: str | Path, *, capture: str | None = None) -> None:
        self._test_sources[str(random_id)] = Path(path)
        if capture is not None:
            self._test_captures[str(random_id)] = str(capture)

    def resolve(self, random_id: str) -> CandidateSourceResolution:
        random_id = str(random_id)
        test_path = self._test_sources.get(random_id)
        if test_path is not None:
            return self._candidate_path_result(
                status=SOURCE_TEST_FIXTURE,
                path=test_path,
                capture=self._test_captures.get(random_id),
                source_kind=SOURCE_TEST_FIXTURE,
            )

        resolved = self._resolve_demo_catalog_source(random_id)
        if resolved.status != SOURCE_NO_CANDIDATE_SOURCE:
            return resolved

        resolved = self._resolve_browser_catalog_source(random_id)
        if resolved.status != SOURCE_NO_CANDIDATE_SOURCE:
            return resolved

        return CandidateSourceResolution(
            status=SOURCE_NO_CANDIDATE_SOURCE,
            source_kind=SOURCE_NO_CANDIDATE_SOURCE,
        )

    def _resolve_demo_catalog_source(self, random_id: str) -> CandidateSourceResolution:
        try:
            records = load_catalog_identify_demo_identity_records()
        except CatalogApiError as exc:
            return CandidateSourceResolution(
                status=SOURCE_NO_CANDIDATE_SOURCE,
                source_kind=SOURCE_DEMO_CATALOG,
                detail=type(exc).__name__,
            )

        for record in records:
            public = record.public_item
            if _demo_random_id(public.dataset, public.id) != random_id:
                continue
            return self._candidate_path_result(
                status=SOURCE_DEMO_CATALOG,
                path=record.enrollment_asset_path,
                capture=record.enrollment_capture,
                source_kind=SOURCE_DEMO_CATALOG,
            )
        return CandidateSourceResolution(
            status=SOURCE_NO_CANDIDATE_SOURCE,
            source_kind=SOURCE_DEMO_CATALOG,
        )

    def _resolve_browser_catalog_source(self, random_id: str) -> CandidateSourceResolution:
        try:
            records = load_catalog_identify_seed_records()
        except CatalogApiError as exc:
            return CandidateSourceResolution(
                status=SOURCE_NO_CANDIDATE_SOURCE,
                source_kind=SOURCE_BROWSER_CATALOG,
                detail=type(exc).__name__,
            )

        for record in records:
            public = record.public_item
            if _browser_random_id(public.dataset, public.identity_id) != random_id:
                continue
            return self._candidate_path_result(
                status=SOURCE_BROWSER_CATALOG,
                path=record.enrollment_asset_path,
                capture=record.enrollment_capture,
                source_kind=SOURCE_BROWSER_CATALOG,
            )
        return CandidateSourceResolution(
            status=SOURCE_NO_CANDIDATE_SOURCE,
            source_kind=SOURCE_BROWSER_CATALOG,
        )

    @staticmethod
    def _candidate_path_result(
        *,
        status: str,
        path: str | Path,
        capture: str | None,
        source_kind: str,
    ) -> CandidateSourceResolution:
        candidate_path = Path(path)
        if not candidate_path.is_file():
            return CandidateSourceResolution(
                status=SOURCE_CANDIDATE_SOURCE_MISSING,
                source_kind=source_kind,
                detail="candidate source file is missing",
            )
        return CandidateSourceResolution(
            status=status,
            path=candidate_path,
            capture=capture,
            source_kind=source_kind,
        )
