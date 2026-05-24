from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


FALLBACK_CANONICAL_BENCHMARK_METHODS = ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"]
FALLBACK_RESEARCH_BENCHMARK_METHODS = ["sift_plain_roll_v2", "dedicated"]
PROFILE_NAMES = ("canonical", "dedicated", "research")


@dataclass(frozen=True)
class BenchmarkMethodProfiles:
    canonical: tuple[str, ...]
    research_methods: tuple[str, ...]
    dedicated: tuple[str, ...]
    loaded_from_registry: bool
    fallback_reason: str | None = None

    @property
    def research(self) -> tuple[str, ...]:
        return (*self.canonical, *self.research_methods)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "canonical": list(self.canonical),
            "research": list(self.research),
            "dedicated": list(self.dedicated),
        }


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _ensure_project_root_on_path() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _fallback_profiles(reason: str | None = None) -> BenchmarkMethodProfiles:
    return BenchmarkMethodProfiles(
        canonical=tuple(FALLBACK_CANONICAL_BENCHMARK_METHODS),
        research_methods=tuple(FALLBACK_RESEARCH_BENCHMARK_METHODS),
        dedicated=("dedicated",),
        loaded_from_registry=False,
        fallback_reason=reason,
    )


@lru_cache(maxsize=1)
def load_benchmark_method_profiles() -> BenchmarkMethodProfiles:
    """Load benchmark method profiles from configs/methods.yaml via the API registry.

    CLI scripts may be invoked directly as file paths, which can leave the
    repository root off sys.path. We add it here and keep a conservative fallback
    so --help and legacy wrappers remain usable if registry imports are broken.
    """

    try:
        _ensure_project_root_on_path()
        from apps.api.method_registry import load_api_method_registry

        registry = load_api_method_registry()
        definitions = registry.list_methods()

        canonical = tuple(
            definition.benchmark_name
            for definition in definitions
            if definition.benchmark_name
            and (definition.benchmark_default or definition.canonical_default)
            and not definition.research_track
        )
        research_methods = tuple(
            definition.benchmark_name
            for definition in definitions
            if definition.benchmark_name and definition.research_track
        )
        dedicated = tuple(
            definition.benchmark_name
            for definition in definitions
            if definition.benchmark_name
            and (definition.canonical_api_name == "dedicated" or definition.benchmark_name == "dedicated")
        )

        if not canonical:
            return _fallback_profiles("registry returned no canonical/default benchmark methods")
        if not dedicated:
            dedicated = ("dedicated",)

        return BenchmarkMethodProfiles(
            canonical=canonical,
            research_methods=research_methods,
            dedicated=dedicated,
            loaded_from_registry=True,
        )
    except Exception as exc:
        return _fallback_profiles(str(exc))


def benchmark_method_profile_map() -> Dict[str, List[str]]:
    return load_benchmark_method_profiles().as_dict()


def canonical_benchmark_methods() -> List[str]:
    return list(load_benchmark_method_profiles().canonical)


def research_benchmark_methods() -> List[str]:
    return list(load_benchmark_method_profiles().research_methods)


def default_benchmark_methods_csv() -> str:
    return ",".join(canonical_benchmark_methods())
