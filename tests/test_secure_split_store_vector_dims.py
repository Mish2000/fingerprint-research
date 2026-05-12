from __future__ import annotations

import numpy as np
import pytest

from apps.api.method_registry import load_api_method_registry
import src.fpbench.identification.secure_split_store as secure_store_module
from src.fpbench.identification.secure_split_store import SecureSplitFingerprintStore, VECTOR_SPECS, VectorSpec

CLASSIC_GENERIC_ONLY_METHODS = ["classic_gftt_orb", "classic_orb", "harris", "minutiae", "sift"]
ALL_GENERIC_STORAGE_METHODS = ["classic_gftt_orb", "classic_orb", "dl", "harris", "minutiae", "sift", "vit"]


@pytest.fixture()
def store() -> SecureSplitFingerprintStore:
    return SecureSplitFingerprintStore.for_inspection(
        database_url="postgresql://user:pass@localhost:5432/biometric_test",
        identity_database_url="postgresql://user:pass@localhost:5433/identity_test",
    )


@pytest.mark.parametrize("dim", [128, 2048])
def test_dl_vector_requires_exact_512_dimensions(store: SecureSplitFingerprintStore, dim: int) -> None:
    with pytest.raises(ValueError, match=rf"vector for method='dl' has dim={dim}, expected exactly 512"):
        store._prepare_vector("dl", np.ones(dim, dtype=np.float32))


def test_dl_vector_accepts_exact_512_and_normalizes(store: SecureSplitFingerprintStore) -> None:
    vec = store._prepare_vector("dl", np.ones(512, dtype=np.float32))

    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    assert norm == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("dim", [512, 128])
def test_vit_vector_requires_exact_768_dimensions(store: SecureSplitFingerprintStore, dim: int) -> None:
    with pytest.raises(ValueError, match=rf"vector for method='vit' has dim={dim}, expected exactly 768"):
        store._prepare_vector("vit", np.ones(dim, dtype=np.float32))


def test_vit_vector_accepts_exact_768_and_normalizes(store: SecureSplitFingerprintStore) -> None:
    vec = store._prepare_vector("vit", np.ones(768, dtype=np.float32))

    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_store_vector_specs_are_derived_from_method_registry() -> None:
    registry_specs = load_api_method_registry().retrieval_vector_specs()

    assert set(VECTOR_SPECS) == set(registry_specs) == {
        "classic_orb",
        "classic_gftt_orb",
        "minutiae",
        "harris",
        "sift",
        "dl",
        "vit",
    }
    for method, store_spec in VECTOR_SPECS.items():
        registry_spec = registry_specs[method]
        assert store_spec.dim == registry_spec.dim
        assert store_spec.vector_kind == registry_spec.vector_kind
        assert store_spec.distance_metric == registry_spec.distance_metric


def test_rerank_only_methods_report_capability_aware_retrieval_error(
    store: SecureSplitFingerprintStore,
) -> None:
    with pytest.raises(ValueError, match="experimental rerank-only method.*validated fixed-size"):
        store._prepare_vector("dedicated", np.ones(128, dtype=np.float32))


def test_store_reports_generic_vector_storage_schema() -> None:
    metadata = SecureSplitFingerprintStore.vector_storage_schema_metadata()

    assert metadata["mode"] == "method_generic_pgvector_table_with_legacy_compat"
    assert metadata["method_generic_vectors_supported"] is True
    assert metadata["schema_accepts_method_generic_vectors"] is True
    assert metadata["generic_primary_key"] == ["random_id", "method", "vector_kind"]
    assert metadata["indexed_dimensions_available"] == [512, 768]
    for method in CLASSIC_GENERIC_ONLY_METHODS:
        assert metadata["configured_vector_specs"][method]["generic_storage_column"] == "vector_512"
        assert metadata["configured_vector_specs"][method]["legacy_storage_column"] is None
        assert metadata["configured_vector_specs"][method]["legacy_storage_enabled"] is False
        assert metadata["configured_vector_specs"][method]["generic_storage_enabled"] is True
        assert metadata["configured_vector_specs"][method]["preferred_storage"] == "generic"
    assert metadata["configured_vector_specs"]["dl"]["generic_storage_column"] == "vector_512"
    assert metadata["configured_vector_specs"]["vit"]["generic_storage_column"] == "vector_768"
    assert metadata["configured_vector_specs"]["dl"]["legacy_storage_column"] == "vector_512"
    assert metadata["legacy_compatibility_methods"] == ["dl", "vit"]
    assert metadata["dual_write_methods"] == ["dl", "vit"]
    assert metadata["generic_storage_methods"] == ALL_GENERIC_STORAGE_METHODS
    assert metadata["generic_only_methods"] == CLASSIC_GENERIC_ONLY_METHODS
    assert metadata["configured_vector_specs"]["dl"]["legacy_storage_enabled"] is True
    assert metadata["configured_vector_specs"]["dl"]["generic_storage_enabled"] is True
    assert metadata["configured_vector_specs"]["dl"]["preferred_storage"] == "dual"


def test_vector_storage_schema_metadata_reports_generic_only_strategy(monkeypatch) -> None:
    specs = dict(secure_store_module.VECTOR_SPECS)
    specs["future"] = VectorSpec(
        method="future",
        dim=512,
        vector_kind="future_embedding",
        distance_metric="cosine",
        generic_storage_column="vector_512",
        legacy_storage_column=None,
        legacy_storage_enabled=False,
        generic_storage_enabled=True,
        preferred_storage="generic",
    )
    monkeypatch.setattr(secure_store_module, "VECTOR_SPECS", specs)

    metadata = SecureSplitFingerprintStore.vector_storage_schema_metadata()

    assert metadata["legacy_compatibility_methods"] == ["dl", "vit"]
    assert metadata["dual_write_methods"] == ["dl", "vit"]
    assert metadata["generic_storage_methods"] == ["classic_gftt_orb", "classic_orb", "dl", "future", "harris", "minutiae", "sift", "vit"]
    assert metadata["generic_only_methods"] == ["classic_gftt_orb", "classic_orb", "future", "harris", "minutiae", "sift"]
    assert metadata["configured_vector_specs"]["future"]["legacy_storage_enabled"] is False
    assert metadata["configured_vector_specs"]["future"]["generic_storage_enabled"] is True
    assert metadata["configured_vector_specs"]["future"]["preferred_storage"] == "generic"
    assert metadata["configured_vector_specs"]["future"]["legacy_storage_column"] is None
    assert metadata["configured_vector_specs"]["future"]["generic_storage_column"] == "vector_512"


def test_generic_vector_check_constraint_is_not_method_specific() -> None:
    check_sql = SecureSplitFingerprintStore._generic_vector_check_constraint_sql()

    assert "method =" not in check_sql
    for method in ("classic_orb", "classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"):
        assert method not in check_sql
    assert "dim = 512" in check_sql
    assert "dim = 768" in check_sql


def test_legacy_vector_check_constraint_only_names_legacy_compat_methods(monkeypatch) -> None:
    specs = dict(secure_store_module.VECTOR_SPECS)
    specs["future"] = VectorSpec(
        method="future",
        dim=512,
        vector_kind="future_embedding",
        distance_metric="cosine",
        generic_storage_column="vector_512",
        legacy_storage_column=None,
        legacy_storage_enabled=False,
        generic_storage_enabled=True,
        preferred_storage="generic",
    )
    monkeypatch.setattr(secure_store_module, "VECTOR_SPECS", specs)

    check_sql = SecureSplitFingerprintStore._legacy_vector_check_constraint_sql()

    assert "method = 'dl'" in check_sql
    assert "method = 'vit'" in check_sql
    for method in ("classic_orb", "classic_gftt_orb", "minutiae", "harris", "sift"):
        assert method not in check_sql
    assert "future" not in check_sql
