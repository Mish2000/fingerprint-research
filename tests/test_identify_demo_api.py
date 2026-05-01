from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile

import apps.api.catalog_store as catalog_store
import apps.api.demo_store as demo_store
import apps.api.main as api_main
from apps.api.identification_service import IdentificationService
from apps.api.main import (
    catalog_identify_gallery,
    identify_browser_reset,
    identify_browser_seed_selection,
    identify_demo_reset,
    identify_demo_seed,
    identify_search,
    identify_stats,
)
from apps.api.schemas import IdentifyBrowserSeedSelectionRequest, MatchMethod
from src.fpbench.identification.secure_split_store import EnrollmentReceipt, PersonDirectoryRecord, RawFingerprintRecord


def _safe_capture(raw: str | None) -> str:
    value = str(raw or "plain").strip().lower()
    return value or "plain"


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_national_id(national_id: str) -> str:
    return "".join(ch for ch in str(national_id).strip() if ch.isdigit())


class InMemoryStore:
    def __init__(self):
        self.people: dict[str, PersonDirectoryRecord] = {}
        self.raw: dict[str, RawFingerprintRecord] = {}
        self.vectors: dict[tuple[str, str], np.ndarray] = {}

    def enroll(
        self,
        *,
        full_name: str,
        national_id: str,
        image_bytes: bytes,
        capture: str,
        ext: str,
        vectors: dict[str, np.ndarray],
        random_id: str | None = None,
        created_at: str | None = None,
        replace_existing: bool = False,
    ) -> EnrollmentReceipt:
        national_id_norm = _normalize_national_id(national_id)
        existing_random_id = None
        for record in self.people.values():
            if record.national_id == national_id_norm:
                existing_random_id = record.random_id
                break
        if existing_random_id is not None and not replace_existing:
            raise ValueError("national_id already enrolled; pass replace_existing=True to rotate the template")
        if existing_random_id is not None:
            self.purge(existing_random_id)

        rid = random_id or uuid.uuid4().hex
        created = created_at or "2026-04-02T00:00:00+00:00"
        self.people[rid] = PersonDirectoryRecord(
            random_id=rid,
            full_name=full_name,
            name_norm=_normalize_name(full_name),
            national_id=national_id_norm,
            created_at=created,
        )
        self.raw[rid] = RawFingerprintRecord(
            random_id=rid,
            capture=_safe_capture(capture),
            ext=ext,
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            created_at=created,
            image_bytes=image_bytes,
        )
        for method, vector in vectors.items():
            self.vectors[(rid, method)] = np.asarray(vector, dtype=np.float32).reshape(-1)
        return EnrollmentReceipt(
            random_id=rid,
            created_at=created,
            vector_methods=sorted(vectors.keys()),
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        )

    def purge(self, random_id: str) -> bool:
        existed = random_id in self.people
        self.people.pop(random_id, None)
        self.raw.pop(random_id, None)
        for key in [key for key in list(self.vectors.keys()) if key[0] == random_id]:
            self.vectors.pop(key, None)
        return existed

    def total_people(self) -> int:
        return len(self.people)

    def count_vectors(self, method: str) -> int:
        return sum(1 for _, vector_method in self.vectors if vector_method == method)

    def search_people(self, hints, *, limit: int | None = None):
        rows = list(self.people.values())
        rows.sort(key=lambda item: item.created_at, reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def get_person(self, random_id: str):
        return self.people.get(random_id)

    def load_raw_fingerprint(self, random_id: str):
        return self.raw.get(random_id)

    def shortlist_by_vector(self, *, method: str, probe_vector: np.ndarray, limit: int, candidate_ids=None):
        probe = np.asarray(probe_vector, dtype=np.float32).reshape(-1)
        ids = list(candidate_ids) if candidate_ids is not None else [rid for rid, m in self.vectors if m == method]
        rows = []
        for rid in ids:
            vector = self.vectors.get((rid, method))
            if vector is None:
                continue
            rows.append((rid, float(np.dot(probe, vector))))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:limit]

    def dump_layout(self):
        return {
            "backend": "memory",
            "dual_database_enabled": "false",
            "person_table": "memory.person_directory",
            "raw_fingerprints_table": "memory.raw_fingerprints",
            "feature_vectors_table": "memory.feature_vectors",
            "identity_map_table": "memory.identity_map",
        }


def _fake_vectorizer(path: str, capture: str | None = None) -> np.ndarray:
    first_byte = Path(path).read_bytes()[:1]
    mapping = {
        b"A": np.array([1.0, 0.0], dtype=np.float32),
        b"B": np.array([0.0, 1.0], dtype=np.float32),
        b"C": np.array([0.7, 0.7], dtype=np.float32),
    }
    return mapping.get(first_byte, np.array([0.2, 0.2], dtype=np.float32))


def _fake_rerank(method: MatchMethod, probe_path: str, candidate_path: str, probe_capture: str, candidate_capture: str) -> float:
    probe = Path(probe_path).read_bytes()[:1]
    candidate = Path(candidate_path).read_bytes()[:1]
    return 0.95 if probe == candidate else 0.05


def _write_file(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _catalog_asset(asset_id: str, *, dataset: str, source_relative_path: str, capture: str, finger: str, subject_id: int) -> dict:
    return {
        "asset_id": asset_id,
        "dataset": dataset,
        "path": f"data/samples/assets/{dataset}/{asset_id}.png",
        "relative_path": f"data/samples/assets/{dataset}/{asset_id}.png",
        "source_path": str((Path("C:/demo-root") / source_relative_path).as_posix()),
        "source_relative_path": source_relative_path,
        "signature": source_relative_path.lower(),
        "subject_id": subject_id,
        "capture": capture,
        "finger": finger,
        "split": "val",
        "availability_status": "available",
        "materialized_asset_kind": "binary_image",
        "recommended_usage": "recommended_enrollment",
        "traceability": {
            "source_dataset": dataset,
            "source_path_signature": source_relative_path.lower(),
            "source_subject_id": subject_id,
            "source_split": "val",
        },
    }


def _identity_payload(
    identity_id: str,
    *,
    dataset: str,
    display_name: str,
    subject_id: int,
    enrollment_asset: dict,
    probe_asset: dict,
) -> dict:
    return {
        "identity_id": identity_id,
        "dataset": dataset,
        "display_name": display_name,
        "subject_id": subject_id,
        "gallery_role": "standard",
        "enrollment_candidates": [enrollment_asset["asset_id"]],
        "probe_candidates": [probe_asset["asset_id"]],
        "recommended_enrollment_asset_id": enrollment_asset["asset_id"],
        "recommended_probe_asset_id": probe_asset["asset_id"],
        "tags": [dataset, "gallery"],
        "is_demo_safe": True,
        "exemplars": [enrollment_asset, probe_asset],
    }


def _scenario_payload(
    scenario_id: str,
    *,
    dataset: str,
    title: str,
    description: str,
    identity_id: str,
    probe_asset: dict,
    recommended_method: str = "dl",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "scenario_type": "positive_identification",
        "dataset": dataset,
        "title": title,
        "description": description,
        "enrollment_identity_ids": [identity_id],
        "probe_asset": probe_asset,
        "expected_identity_id": identity_id,
        "difficulty": "easy",
        "recommended_method": recommended_method,
        "tags": [dataset, "demo"],
        "is_demo_safe": True,
    }


def _browser_item(
    asset_id: str,
    *,
    dataset: str,
    source_relative_path: str,
    thumbnail_path: str,
    preview_path: str,
    subject_id: str,
    finger: str,
    capture: str,
) -> dict:
    return {
        "asset_id": asset_id,
        "dataset": dataset,
        "split": "val",
        "source_path": source_relative_path,
        "thumbnail_path": thumbnail_path,
        "preview_path": preview_path,
        "availability_status": "available",
        "subject_id": subject_id,
        "finger": finger,
        "capture": capture,
        "modality": "optical_2d",
        "ui_eligible": True,
        "selection_reason": "Deterministic identify-demo preview.",
        "selection_policy": "deterministic_round_robin",
        "original_dimensions": {"width": 640, "height": 480},
        "thumbnail_dimensions": {"width": 160, "height": 120},
        "preview_dimensions": {"width": 512, "height": 384},
        "traceability": {
            "source_path_relative": source_relative_path,
            "source_path_original": str((Path("C:/demo-root") / source_relative_path).as_posix()),
            "thumbnail_path": thumbnail_path,
            "preview_path": preview_path,
        },
    }


def _configure_roots(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    for module in (demo_store, catalog_store):
        monkeypatch.setattr(module, "ROOT", repo_root)
        monkeypatch.setattr(module, "SAMPLES_ROOT", repo_root / "data" / "samples")
        monkeypatch.setattr(module, "CATALOG_PATH", repo_root / "data" / "samples" / "catalog.json")
        monkeypatch.setattr(module, "ASSETS_ROOT", repo_root / "data" / "samples" / "assets")
    monkeypatch.setattr(catalog_store, "PROCESSED_ROOT", repo_root / "data" / "processed")
    monkeypatch.setattr(catalog_store, "UI_ASSETS_REGISTRY_PATH", repo_root / "data" / "processed" / "ui_assets_registry.json")
    monkeypatch.setenv("FPBENCH_IDENTIFY_DEMO_STATE_PATH", str(repo_root / "artifacts" / "runtime" / "identify_demo_store_state.json"))
    monkeypatch.setenv("FPBENCH_IDENTIFY_BROWSER_STATE_PATH", str(repo_root / "artifacts" / "runtime" / "identify_browser_store_state.json"))
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()


def _build_identify_demo_artifacts(repo_root: Path) -> None:
    dataset = "identify_demo"
    source_one = "data/raw/identify_demo/subject_001_probe.png"
    source_two = "data/raw/identify_demo/subject_002_probe.png"

    enroll_one = _catalog_asset("asset_subject_001_enroll", dataset=dataset, source_relative_path=source_one, capture="plain", finger="1", subject_id=1)
    probe_one = _catalog_asset("asset_subject_001_probe", dataset=dataset, source_relative_path=source_one, capture="roll", finger="1", subject_id=1)
    enroll_two = _catalog_asset("asset_subject_002_enroll", dataset=dataset, source_relative_path=source_two, capture="plain", finger="2", subject_id=2)
    probe_two = _catalog_asset("asset_subject_002_probe", dataset=dataset, source_relative_path=source_two, capture="roll", finger="2", subject_id=2)

    catalog_payload = {
        "catalog_version": "1.0.0",
        "generated_at": "2026-04-02T00:00:00Z",
        "source_datasets": [
            {"dataset": dataset, "dataset_label": "Identify Demo"},
        ],
        "verify_cases": [],
        "identify_gallery": {
            "identities": [
                _identity_payload("identity_subject_001", dataset=dataset, display_name="Subject 001", subject_id=1, enrollment_asset=enroll_one, probe_asset=probe_one),
                _identity_payload("identity_subject_002", dataset=dataset, display_name="Subject 002", subject_id=2, enrollment_asset=enroll_two, probe_asset=probe_two),
            ],
            "demo_scenarios": [
                _scenario_payload(
                    "probe_subject_001",
                    dataset=dataset,
                    title="Positive identify probe",
                    description="Probe the seeded gallery with subject 001.",
                    identity_id="identity_subject_001",
                    probe_asset=probe_one,
                    recommended_method="dl",
                )
            ],
        },
    }
    _write_json(repo_root / "data" / "samples" / "catalog.json", catalog_payload)
    _write_file(repo_root / enroll_one["relative_path"], b"A_catalog")
    _write_file(repo_root / probe_one["relative_path"], b"A_catalog_probe")
    _write_file(repo_root / enroll_two["relative_path"], b"B_catalog")
    _write_file(repo_root / probe_two["relative_path"], b"B_catalog_probe")

    preview_one = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "previews" / "uiasset_subject_001.png", b"A_preview")
    thumb_one = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "thumbnails" / "uiasset_subject_001.png", b"A_thumb")
    preview_two = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "previews" / "uiasset_subject_002.png", b"B_preview")
    thumb_two = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "thumbnails" / "uiasset_subject_002.png", b"B_thumb")

    browser_index = {
        "dataset": dataset,
        "dataset_label": "Identify Demo",
        "generated_at": "2026-04-02T00:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "validation_status": "pass",
        "summary": {"items_generated": 2},
        "items": [
            _browser_item(
                "uiasset_subject_001",
                dataset=dataset,
                source_relative_path=source_one,
                thumbnail_path=thumb_one.relative_to(repo_root).as_posix(),
                preview_path=preview_one.relative_to(repo_root).as_posix(),
                subject_id="1",
                finger="1",
                capture="plain",
            ),
            _browser_item(
                "uiasset_subject_002",
                dataset=dataset,
                source_relative_path=source_two,
                thumbnail_path=thumb_two.relative_to(repo_root).as_posix(),
                preview_path=preview_two.relative_to(repo_root).as_posix(),
                subject_id="2",
                finger="2",
                capture="plain",
            ),
        ],
    }
    validation_report = {
        "dataset": dataset,
        "dataset_label": "Identify Demo",
        "generated_at": "2026-04-02T00:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "source_records_checked": 2,
        "generated_items": 2,
        "excluded_records": 0,
        "missing_source_files": 0,
        "unreadable_source_files": 0,
        "missing_critical_metadata": 0,
        "duplicates_skipped": 0,
        "validation_status": "pass",
    }
    _write_json(repo_root / "data" / "processed" / dataset / "ui_assets" / "index.json", browser_index)
    _write_json(repo_root / "data" / "processed" / dataset / "ui_assets" / "validation_report.json", validation_report)
    _write_json(
        repo_root / "data" / "processed" / "ui_assets_registry.json",
        {
            "generated_at": "2026-04-02T00:00:00Z",
            "generator_version": "1.0.0",
            "datasets": [
                {
                    "dataset": dataset,
                    "index_path": "data/processed/identify_demo/ui_assets/index.json",
                    "validation_report_path": "data/processed/identify_demo/ui_assets/validation_report.json",
                    "item_count": 2,
                    "validation_status": "pass",
                    "selection_policy": "deterministic_round_robin",
                    "summary": {"items_generated": 2},
                }
            ],
        },
    )


def _append_alt_identify_dataset_artifacts(repo_root: Path) -> None:
    dataset = "identify_demo_alt"
    source_main = "data/raw/identify_demo_alt/subject_101_probe.png"
    source_followup = "data/raw/identify_demo_alt/subject_101_followup.png"
    source_extra = "data/raw/identify_demo_alt/subject_101_extra.png"

    enroll = _catalog_asset("asset_alt_subject_101_enroll", dataset=dataset, source_relative_path=source_main, capture="plain", finger="3", subject_id=101)
    probe = _catalog_asset("asset_alt_subject_101_probe", dataset=dataset, source_relative_path=source_followup, capture="latent", finger="3", subject_id=101)

    catalog_path = repo_root / "data" / "samples" / "catalog.json"
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_payload["source_datasets"].append({"dataset": dataset, "dataset_label": "Identify Demo Alt"})
    catalog_payload["identify_gallery"]["identities"].append(
        _identity_payload(
            "identity_alt_subject_101",
            dataset=dataset,
            display_name="Subject 101",
            subject_id=101,
            enrollment_asset=enroll,
            probe_asset=probe,
        )
    )
    _write_json(catalog_path, catalog_payload)

    _write_file(repo_root / enroll["relative_path"], b"C_catalog")
    _write_file(repo_root / probe["relative_path"], b"C_catalog_followup")
    _write_file(repo_root / source_extra, b"C_catalog_extra")

    preview_main = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "previews" / "uiasset_subject_101.png", b"C_preview")
    thumb_main = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "thumbnails" / "uiasset_subject_101.png", b"C_thumb")
    preview_followup = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "previews" / "uiasset_subject_101_followup.png", b"C_followup_preview")
    thumb_followup = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "thumbnails" / "uiasset_subject_101_followup.png", b"C_followup_thumb")
    preview_extra = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "previews" / "uiasset_subject_101_extra.png", b"C_extra_preview")
    thumb_extra = _write_file(repo_root / "data" / "processed" / dataset / "ui_assets" / "thumbnails" / "uiasset_subject_101_extra.png", b"C_extra_thumb")

    browser_index = {
        "dataset": dataset,
        "dataset_label": "Identify Demo Alt",
        "generated_at": "2026-04-02T00:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "validation_status": "pass",
        "summary": {"items_generated": 3},
        "items": [
            _browser_item(
                "uiasset_subject_101",
                dataset=dataset,
                source_relative_path=source_main,
                thumbnail_path=thumb_main.relative_to(repo_root).as_posix(),
                preview_path=preview_main.relative_to(repo_root).as_posix(),
                subject_id="101",
                finger="3",
                capture="plain",
            ),
            _browser_item(
                "uiasset_subject_101_followup",
                dataset=dataset,
                source_relative_path=source_followup,
                thumbnail_path=thumb_followup.relative_to(repo_root).as_posix(),
                preview_path=preview_followup.relative_to(repo_root).as_posix(),
                subject_id="101",
                finger="3",
                capture="latent",
            ),
            _browser_item(
                "uiasset_subject_101_extra",
                dataset=dataset,
                source_relative_path=source_extra,
                thumbnail_path=thumb_extra.relative_to(repo_root).as_posix(),
                preview_path=preview_extra.relative_to(repo_root).as_posix(),
                subject_id="101",
                finger="3",
                capture="latent",
            ),
        ],
    }
    validation_report = {
        "dataset": dataset,
        "dataset_label": "Identify Demo Alt",
        "generated_at": "2026-04-02T00:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "source_records_checked": 3,
        "generated_items": 3,
        "excluded_records": 0,
        "missing_source_files": 0,
        "unreadable_source_files": 0,
        "missing_critical_metadata": 0,
        "duplicates_skipped": 0,
        "validation_status": "pass",
    }
    _write_json(repo_root / "data" / "processed" / dataset / "ui_assets" / "index.json", browser_index)
    _write_json(repo_root / "data" / "processed" / dataset / "ui_assets" / "validation_report.json", validation_report)

    registry_path = repo_root / "data" / "processed" / "ui_assets_registry.json"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["datasets"].append(
        {
            "dataset": dataset,
            "index_path": "data/processed/identify_demo_alt/ui_assets/index.json",
            "validation_report_path": "data/processed/identify_demo_alt/ui_assets/validation_report.json",
            "item_count": 3,
            "validation_status": "pass",
            "selection_policy": "deterministic_round_robin",
            "summary": {"items_generated": 3},
        }
    )
    _write_json(registry_path, registry_payload)


def _append_unsafe_identify_scenario(repo_root: Path) -> None:
    catalog_path = repo_root / "data" / "samples" / "catalog.json"
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    safe_probe_asset = catalog_payload["identify_gallery"]["demo_scenarios"][0]["probe_asset"]
    catalog_payload["identify_gallery"]["demo_scenarios"].append(
        {
            "scenario_id": "unsafe_hidden_probe",
            "scenario_type": "no_match",
            "dataset": "identify_demo",
            "title": "Unsafe hidden probe",
            "description": "This entry should not be exposed by the demo-safe API layer.",
            "enrollment_identity_ids": ["identity_subject_001"],
            "probe_asset": safe_probe_asset,
            "expected_identity_id": None,
            "difficulty": "hard",
            "recommended_method": "dl",
            "tags": ["unsafe", "hidden"],
            "is_demo_safe": False,
        }
    )
    _write_json(catalog_path, catalog_payload)


@pytest.fixture(autouse=True)
def _clear_caches_and_service():
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()
    original_service = api_main._ident_service
    original_browser_service = api_main._browser_ident_service
    yield
    api_main._ident_service = original_service
    api_main._browser_ident_service = original_browser_service
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()


def _install_service() -> IdentificationService:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )
    browser_service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )
    api_main._ident_service = service
    api_main._browser_ident_service = browser_service
    return service


def test_catalog_identify_gallery_exposes_demo_identities_and_probe_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)

    response = catalog_identify_gallery(dataset="identify_demo", limit=10, offset=0)

    assert response.total == 2
    assert len(response.items) == 2
    assert len(response.demo_identities) == 2
    assert len(response.probe_cases) >= 1
    assert response.demo_identities[0].thumbnail_url.startswith("/api/catalog/assets/identify_demo/")
    scenario_probe = next(
        probe_case
        for probe_case in response.probe_cases
        if probe_case.id == "probe_subject_001"
    )
    assert scenario_probe.probe_asset_url == scenario_probe.probe_preview_url
    assert scenario_probe.expected_top_identity_label == "Subject 001"
    assert scenario_probe.recommended_retrieval_method == "dl"
    assert response.total_probe_cases == len(response.probe_cases)


def test_identify_demo_seed_is_idempotent_and_stats_track_demo_seeded_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    _install_service()

    first = identify_demo_seed()
    second = identify_demo_seed()
    stats = identify_stats()

    assert first.seeded_count == 2
    assert first.updated_count == 0
    assert first.total_enrolled == 2
    assert first.demo_seeded_count == 2

    assert second.seeded_count == 0
    assert second.updated_count == 2
    assert second.total_enrolled == 2
    assert second.demo_seeded_count == 2

    assert stats.total_enrolled == 2
    assert stats.demo_seeded_count == 2
    assert stats.browser_seeded_count == 0


def test_identify_demo_reset_only_removes_demo_seeded_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    service = _install_service()

    manual_probe = _write_file(repo_root / "manual_identity.bin", b"C_manual")
    manual_receipt = service.enroll_from_path(
        path=str(manual_probe),
        full_name="Manual Person",
        national_id="123456789",
        capture="plain",
        vector_methods=("dl",),
    )

    seed_response = identify_demo_seed()
    reset_response = identify_demo_reset()
    stats = identify_stats()

    assert seed_response.total_enrolled == 3
    assert reset_response.removed_count == 2
    assert reset_response.total_enrolled == 1
    assert reset_response.demo_seeded_count == 0
    assert stats.total_enrolled == 1
    assert stats.demo_seeded_count == 0
    assert stats.browser_seeded_count == 0
    assert service.store.get_person(manual_receipt.random_id) is not None


def test_catalog_identify_gallery_filters_by_dataset_and_keeps_demo_probe_order_demo_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _append_alt_identify_dataset_artifacts(repo_root)
    _append_unsafe_identify_scenario(repo_root)
    _configure_roots(monkeypatch, repo_root)

    response = catalog_identify_gallery(dataset="identify_demo_alt", limit=10, offset=0)

    assert response.total == 1
    assert all(item.dataset == "identify_demo_alt" for item in response.items)
    assert response.probe_cases
    assert [probe_case.scenario_type for probe_case in response.probe_cases] == [
        "positive_identification",
        "positive_identification",
        "difficult_identification",
    ]
    assert {probe_case.id for probe_case in response.probe_cases} == {
        "probe_identity_alt_subject_101",
        "probe_identity_alt_subject_101_positive_followup",
        "probe_identity_alt_subject_101_harder_positive",
    }

    demo_response = catalog_identify_gallery(dataset="identify_demo", limit=10, offset=0)
    assert all(probe_case.id != "unsafe_hidden_probe" for probe_case in demo_response.probe_cases)


def test_identify_browser_seed_selection_seeds_only_the_requested_identities_into_the_isolated_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    operational_service = _install_service()

    response = identify_browser_seed_selection(
        IdentifyBrowserSeedSelectionRequest(
            dataset="identify_demo",
            selected_identity_ids=["identity_subject_002"],
            overwrite=True,
            metadata={"source": "pytest"},
        )
    )
    stats = identify_stats()

    assert response.selected_count == 1
    assert response.seeded_count == 1
    assert response.browser_seeded_count == 1
    assert response.seeded_identity_ids == ["identity_subject_002"]
    assert response.store_ready is True
    assert response.errors == []
    assert api_main._browser_ident_service is not None
    assert api_main._browser_ident_service.store.get_person("browser_identify_identify_demo_identity_subject_002") is not None
    assert api_main._browser_ident_service.store.get_person("browser_identify_identify_demo_identity_subject_001") is None
    assert operational_service.store.total_people() == 0
    assert stats.total_enrolled == 0
    assert stats.browser_seeded_count == 1


def test_identify_browser_reset_is_isolated_from_operational_enrollments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    service = _install_service()

    manual_probe = _write_file(repo_root / "manual_operational_identity.bin", b"C_manual")
    manual_receipt = service.enroll_from_path(
        path=str(manual_probe),
        full_name="Manual Operational Person",
        national_id="987654321",
        capture="plain",
        vector_methods=("dl",),
    )

    identify_browser_seed_selection(
        IdentifyBrowserSeedSelectionRequest(
            dataset="identify_demo",
            selected_identity_ids=["identity_subject_001", "identity_subject_002"],
            overwrite=True,
        )
    )
    reset_response = identify_browser_reset()
    stats = identify_stats()

    assert reset_response.removed_count == 2
    assert reset_response.browser_seeded_count == 0
    assert api_main._browser_ident_service is not None
    assert api_main._browser_ident_service.store.total_people() == 0
    assert service.store.get_person(manual_receipt.random_id) is not None
    assert stats.total_enrolled == 1
    assert stats.browser_seeded_count == 0


def test_identify_search_can_run_against_the_browser_seeded_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    _install_service()

    identify_browser_seed_selection(
        IdentifyBrowserSeedSelectionRequest(
            dataset="identify_demo",
            selected_identity_ids=["identity_subject_001"],
            overwrite=True,
        )
    )

    upload = UploadFile(filename="browser_probe.png", file=BytesIO(b"A_browser_probe"))
    response = asyncio.run(
        identify_search(
            img=upload,
            capture="plain",
            retrieval_method="dl",
            rerank_method=MatchMethod.sift,
            shortlist_size=5,
            threshold=None,
            name_pattern=None,
            national_id_pattern=None,
            created_from=None,
            created_to=None,
            store_scope="browser",
        )
    )

    assert response.top_candidate is not None
    assert response.top_candidate.full_name == "Subject 001"
    assert response.total_enrolled == 1
    assert response.candidate_pool_size == 1


def test_identify_browser_seed_selection_reports_clear_failures_for_unseedable_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_identify_demo_artifacts(repo_root)
    _configure_roots(monkeypatch, repo_root)
    _install_service()

    response = identify_browser_seed_selection(
        IdentifyBrowserSeedSelectionRequest(
            dataset="identify_demo",
            selected_identity_ids=["missing_identity"],
            overwrite=True,
        )
    )

    assert response.selected_count == 1
    assert response.seeded_count == 0
    assert response.browser_seeded_count == 0
    assert response.store_ready is False
    assert response.seeded_identity_ids == []
    assert response.errors
    assert "No browser-seedable identities" in response.errors[0]
