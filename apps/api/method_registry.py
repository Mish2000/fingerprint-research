from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS_CONFIG_PATH = ROOT / "configs" / "methods.yaml"
THRESHOLDS_CONFIG_PATH = ROOT / "configs" / "thresholds.yaml"


class MethodRegistryError(ValueError):
    """Raised when method metadata cannot be loaded or resolved."""


@dataclass(frozen=True)
class MethodIdentificationRole:
    retrieval_capable: bool
    rerank_capable: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApiMethodDefinition:
    canonical_api_name: str
    accepted_aliases: tuple[str, ...]
    benchmark_name: str
    ui_label: str
    family: str
    status: str
    embedding_dim: int | None
    decision_threshold: float
    runtime_defaults: Dict[str, Any]
    benchmark_defaults: Dict[str, Any]
    notes: tuple[str, ...]
    identification_role: MethodIdentificationRole


@dataclass(frozen=True)
class ResolvedApiMethod:
    definition: ApiMethodDefinition
    requested_name: str
    resolved_from_alias: bool

    @property
    def canonical_api_name(self) -> str:
        return self.definition.canonical_api_name

    @property
    def benchmark_name(self) -> str:
        return self.definition.benchmark_name

    @property
    def ui_label(self) -> str:
        return self.definition.ui_label

    @property
    def family(self) -> str:
        return self.definition.family

    @property
    def status(self) -> str:
        return self.definition.status

    @property
    def decision_threshold(self) -> float:
        return self.definition.decision_threshold

    @property
    def accepted_aliases(self) -> tuple[str, ...]:
        return self.definition.accepted_aliases

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "canonical_method": self.canonical_api_name,
            "requested_method": self.requested_name,
            "benchmark_method": self.benchmark_name,
            "method_label": self.ui_label,
            "family": self.family,
            "status": self.status,
            "embedding_dim": self.definition.embedding_dim,
            "aliases": list(self.accepted_aliases),
            "resolved_from_alias": bool(self.resolved_from_alias),
        }


class ApiMethodRegistry:
    def __init__(self, methods_payload: Mapping[str, Any], thresholds_payload: Mapping[str, Any]):
        self._methods_payload = copy.deepcopy(dict(methods_payload))
        self._thresholds_payload = copy.deepcopy(dict(thresholds_payload))

        namespaces = self._methods_payload.get("namespaces", {})
        self.api_runtime_namespace = tuple(
            str(name).strip().lower()
            for name in namespaces.get("api_runtime", [])
            if str(name).strip()
        )
        self.benchmark_runtime_namespace = tuple(
            str(name).strip().lower()
            for name in namespaces.get("benchmark_runtime", [])
            if str(name).strip()
        )

        identification_cfg = self._methods_payload.get("identification", {})
        self.identification_notes = tuple(
            str(note).strip()
            for note in identification_cfg.get("notes", [])
            if str(note).strip()
        )
        self._retrieval_capable = {
            str(name).strip().lower()
            for name in identification_cfg.get("shortlist_retrieval_methods", [])
            if str(name).strip()
        }
        self._rerank_capable = {
            str(name).strip().lower()
            for name in identification_cfg.get("rerank_methods", [])
            if str(name).strip()
        }

        preprocess_cfg = self._thresholds_payload.get("preprocess", {})
        self.preprocess_defaults = copy.deepcopy(
            preprocess_cfg.get("shared_defaults", {}) or {}
        )

        self._definitions: Dict[str, ApiMethodDefinition] = {}
        self._lookup_to_canonical: Dict[str, str] = {}
        self._benchmark_to_canonical: Dict[str, str] = {}

        methods_section = self._methods_payload.get("methods", {})
        benchmark_names = set(self.benchmark_runtime_namespace)

        for canonical_name in self.api_runtime_namespace:
            payload = methods_section.get(canonical_name)
            if not isinstance(payload, Mapping):
                raise MethodRegistryError(
                    f"Missing method definition for api runtime method {canonical_name!r} in {METHODS_CONFIG_PATH}."
                )

            api_name = str(payload.get("api_name") or canonical_name).strip().lower()
            if api_name != canonical_name:
                raise MethodRegistryError(
                    f"Method {canonical_name!r} declared api_name={api_name!r}; expected the canonical API name."
                )

            benchmark_name = str(payload.get("benchmark_name") or canonical_name).strip().lower()
            aliases: list[str] = []
            for raw_alias in payload.get("accepted_aliases", []) or []:
                alias = str(raw_alias).strip().lower()
                if alias and alias != canonical_name and alias not in aliases:
                    aliases.append(alias)
            if benchmark_name and benchmark_name != canonical_name and benchmark_name in benchmark_names:
                aliases.append(benchmark_name)

            decision_threshold_ref = (
                str(payload.get("scoring", {}).get("decision_threshold_ref") or "").strip()
                or f"decision.api.{canonical_name}"
            )
            decision_threshold = float(self._resolve_threshold(decision_threshold_ref))

            runtime_defaults = copy.deepcopy(payload.get("api_runtime_defaults", {}) or {})
            benchmark_defaults = copy.deepcopy(payload.get("benchmark_defaults", {}) or {})
            notes = tuple(
                str(note).strip()
                for note in payload.get("notes", []) or []
                if str(note).strip()
            )
            raw_embedding_dim = payload.get("embedding_dim")
            embedding_dim = int(raw_embedding_dim) if raw_embedding_dim not in (None, "") else None

            retrieval_capable = canonical_name in self._retrieval_capable
            rerank_capable = canonical_name in self._rerank_capable
            role_notes: list[str] = []
            if retrieval_capable:
                role_notes.append("Supported for 1:N shortlist retrieval.")
            if rerank_capable:
                role_notes.append("Supported for 1:N reranking.")
            if rerank_capable and not retrieval_capable:
                role_notes.append("Rerank-only for 1:N; this method is not used for shortlist retrieval.")

            definition = ApiMethodDefinition(
                canonical_api_name=canonical_name,
                accepted_aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                benchmark_name=benchmark_name,
                ui_label=str(payload.get("ui_label") or canonical_name),
                family=str(payload.get("family") or "unknown"),
                status=str(payload.get("status") or "unknown"),
                embedding_dim=embedding_dim,
                decision_threshold=decision_threshold,
                runtime_defaults=runtime_defaults,
                benchmark_defaults=benchmark_defaults,
                notes=notes,
                identification_role=MethodIdentificationRole(
                    retrieval_capable=retrieval_capable,
                    rerank_capable=rerank_capable,
                    notes=tuple(role_notes),
                ),
            )
            self._definitions[canonical_name] = definition

        for definition in self._definitions.values():
            benchmark_name = str(definition.benchmark_name).strip().lower()
            if benchmark_name:
                existing_benchmark = self._benchmark_to_canonical.get(benchmark_name)
                if existing_benchmark is not None and existing_benchmark != definition.canonical_api_name:
                    raise MethodRegistryError(
                        f"Benchmark method name conflict for {benchmark_name!r}: "
                        f"{existing_benchmark!r} vs {definition.canonical_api_name!r}."
                    )
                self._benchmark_to_canonical[benchmark_name] = definition.canonical_api_name
            for lookup_name in (definition.canonical_api_name, *definition.accepted_aliases):
                existing = self._lookup_to_canonical.get(lookup_name)
                if existing is not None and existing != definition.canonical_api_name:
                    raise MethodRegistryError(
                        f"Method alias conflict for {lookup_name!r}: {existing!r} vs {definition.canonical_api_name!r}."
                    )
                self._lookup_to_canonical[lookup_name] = definition.canonical_api_name

    def _resolve_threshold(self, ref: str) -> Any:
        current: Any = self._thresholds_payload
        for part in str(ref).split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise MethodRegistryError(
                    f"Threshold reference {ref!r} could not be resolved in {THRESHOLDS_CONFIG_PATH}."
                )
            current = current[part]
        return current

    def list_methods(self) -> list[ApiMethodDefinition]:
        return [self._definitions[name] for name in self.api_runtime_namespace]

    def definition_for(self, canonical_name: str) -> ApiMethodDefinition:
        key = str(canonical_name).strip().lower()
        if key not in self._definitions:
            raise MethodRegistryError(f"Unknown canonical API method: {canonical_name!r}")
        return self._definitions[key]

    def alias_pairs(self) -> Dict[str, str]:
        pairs: Dict[str, str] = {}
        for definition in self.list_methods():
            for alias in definition.accepted_aliases:
                pairs[alias] = definition.canonical_api_name
        return pairs

    def canonical_name_from_benchmark(self, benchmark_name: Any) -> str | None:
        key = str(benchmark_name or "").strip().lower()
        if not key:
            return None
        return self._benchmark_to_canonical.get(key)

    def definition_from_benchmark(self, benchmark_name: Any) -> ApiMethodDefinition | None:
        canonical_name = self.canonical_name_from_benchmark(benchmark_name)
        if canonical_name is None:
            return None
        return self._definitions[canonical_name]

    def supported_method_names(self) -> tuple[str, ...]:
        return self.api_runtime_namespace

    def supported_retrieval_methods(self) -> tuple[str, ...]:
        return tuple(
            definition.canonical_api_name
            for definition in self.list_methods()
            if definition.identification_role.retrieval_capable
        )

    def supported_rerank_methods(self) -> tuple[str, ...]:
        return tuple(
            definition.canonical_api_name
            for definition in self.list_methods()
            if definition.identification_role.rerank_capable
        )

    def _supported_message(self, *, field_name: str) -> str:
        aliases = self.alias_pairs()
        alias_message = ""
        if aliases:
            rendered = ", ".join(f"{alias} -> {canonical}" for alias, canonical in sorted(aliases.items()))
            alias_message = f" Accepted aliases: {rendered}."
        return (
            f"{field_name} must be one of {list(self.supported_method_names())}.{alias_message}"
        )

    def resolve(self, requested_name: Any, *, field_name: str = "method") -> ResolvedApiMethod:
        raw_value = getattr(requested_name, "value", requested_name)
        requested = str(raw_value or "").strip().lower()
        if not requested:
            raise MethodRegistryError(f"{field_name} is required. {self._supported_message(field_name=field_name)}")

        canonical_name = self._lookup_to_canonical.get(requested)
        if canonical_name is None:
            raise MethodRegistryError(
                f"Unsupported {field_name}={requested!r}. {self._supported_message(field_name=field_name)}"
            )

        definition = self._definitions[canonical_name]
        return ResolvedApiMethod(
            definition=definition,
            requested_name=requested,
            resolved_from_alias=requested != canonical_name,
        )

    def resolve_retrieval_method(self, requested_name: Any) -> ResolvedApiMethod:
        resolved = self.resolve(requested_name, field_name="retrieval_method")
        if not resolved.definition.identification_role.retrieval_capable:
            raise MethodRegistryError(
                f"retrieval_method={resolved.requested_name!r} resolves to {resolved.canonical_api_name!r}, "
                f"which is not configured for 1:N shortlist retrieval. Supported retrieval methods: "
                f"{list(self.supported_retrieval_methods())}."
            )
        return resolved

    def resolve_rerank_method(self, requested_name: Any) -> ResolvedApiMethod:
        resolved = self.resolve(requested_name, field_name="rerank_method")
        if not resolved.definition.identification_role.rerank_capable:
            raise MethodRegistryError(
                f"rerank_method={resolved.requested_name!r} resolves to {resolved.canonical_api_name!r}, "
                f"which is not configured for 1:N reranking. Supported rerank methods: "
                f"{list(self.supported_rerank_methods())}."
            )
        return resolved


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise MethodRegistryError(f"Failed to read method registry config {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise MethodRegistryError(f"Expected a mapping at the root of {path}.")
    return payload


@lru_cache(maxsize=1)
def load_api_method_registry() -> ApiMethodRegistry:
    methods_payload = _load_yaml(METHODS_CONFIG_PATH)
    thresholds_payload = _load_yaml(THRESHOLDS_CONFIG_PATH)
    return ApiMethodRegistry(methods_payload=methods_payload, thresholds_payload=thresholds_payload)


def list_method_catalog_entries() -> Iterable[ApiMethodDefinition]:
    return load_api_method_registry().list_methods()
