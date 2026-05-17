from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET = "nist_sd300b"
DEFAULT_METHODS = ("classic_gftt_orb", "minutiae", "harris", "sift", "dl")
DEFAULT_OUTPUT_DIR = (
    "artifacts/reports/identification/vector_reproducibility_demo"
)
DEFAULT_PREVIEW_DIMS = 40
DEFAULT_EPSILON = 1e-6

PATH_COLUMNS = ("path", "image_path", "file_path", "filepath")
SUMMARY_COLUMNS = (
    "method",
    "vector_dim",
    "exact_equal",
    "allclose_equal",
    "max_abs_diff",
    "mean_abs_diff",
    "nonzero_diff_count",
    "cosine_similarity",
    "binary_equal_dimension_count",
    "total_dimensions",
    "binary_equal_dimension_rate",
    "binary_equal_bit_count",
    "total_binary_bit_count",
    "binary_equal_bit_rate",
    "vector_1_sha256",
    "vector_2_sha256",
)
COMPARISON_COLUMNS = ("dimension", "vector_run_1", "vector_run_2", "abs_diff")
BINARY_COMPARISON_COLUMNS = (
    "dimension",
    "float_run_1",
    "float_run_2",
    "abs_diff",
    "float32_hex_run_1",
    "float32_hex_run_2",
    "float32_binary_run_1",
    "float32_binary_run_2",
    "binary_equal",
)


@dataclass(frozen=True)
class SelectedImage:
    image_path: Path
    dataset: str
    capture: str
    capture_source: str
    source: str
    manifest_path: Path | None = None
    manifest_row_index: int | None = None
    valid_sample_index: int | None = None
    subject_id: str | None = None
    frgp: str | None = None
    impression: str | None = None
    ppi: str | None = None
    split: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "dataset": self.dataset,
            "capture": self.capture,
            "capture_source": self.capture_source,
            "source": self.source,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_row_index": self.manifest_row_index,
            "valid_sample_index": self.valid_sample_index,
            "subject_id": self.subject_id,
            "frgp": self.frgp,
            "impression": self.impression,
            "ppi": self.ppi,
            "split": self.split,
        }


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _infer_capture_from_filename(name: str) -> str | None:
    lowered = str(name or "").lower()
    if "contactless" in lowered or "contact-less" in lowered or "contact_less" in lowered:
        return "contactless"
    if "contact_based" in lowered or "contact-based" in lowered or "contactbased" in lowered:
        return "contact_based"
    if "roll" in lowered or "rolled" in lowered:
        return "roll"
    if "plain" in lowered:
        return "plain"
    return None


def _resolve_repo_path(raw_path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = str(raw_path or "").strip().strip('"')
    if raw.startswith("file:"):
        raw = raw[len("file:") :]
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def default_manifest_for_dataset(dataset: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "manifests" / dataset / "manifest.csv"


def resolve_output_dir(raw_path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    return _resolve_repo_path(raw_path, repo_root=repo_root)


def select_image(
    *,
    image_path: str | None,
    dataset: str,
    manifest_path: str | Path | None,
    sample_index: int,
    capture: str | None,
    repo_root: Path = REPO_ROOT,
) -> SelectedImage:
    capture_cli = _clean_text(capture)
    if image_path:
        resolved_path = _resolve_repo_path(image_path, repo_root=repo_root)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Missing image file: {resolved_path}")
        inferred_capture = _infer_capture_from_filename(resolved_path.name)
        selected_capture = capture_cli or inferred_capture or "plain"
        capture_source = "cli" if capture_cli else ("filename" if inferred_capture else "default_plain")
        return SelectedImage(
            image_path=resolved_path,
            dataset=str(dataset),
            capture=selected_capture,
            capture_source=capture_source,
            source="image_path",
        )

    manifest = (
        _resolve_repo_path(manifest_path, repo_root=repo_root)
        if manifest_path is not None
        else default_manifest_for_dataset(dataset, repo_root=repo_root)
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest}")
    if int(sample_index) < 1:
        raise ValueError("--sample-index is 1-based and must be >= 1")

    valid_rows: list[tuple[int, dict[str, str], Path]] = []
    missing_path_count = 0
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if not any(column in fieldnames for column in PATH_COLUMNS):
            raise ValueError(
                f"Manifest {manifest} must contain one of these image path columns: {list(PATH_COLUMNS)}"
            )
        for row_index, row in enumerate(reader, start=1):
            raw_image_path = next(
                (_clean_text(row.get(column)) for column in PATH_COLUMNS if _clean_text(row.get(column))),
                None,
            )
            if raw_image_path is None:
                missing_path_count += 1
                continue
            resolved_path = _resolve_repo_path(raw_image_path, repo_root=repo_root)
            if not resolved_path.is_file():
                missing_path_count += 1
                continue
            valid_rows.append((row_index, dict(row), resolved_path))

    if not valid_rows:
        raise RuntimeError(
            f"No valid existing image paths found in {manifest}; skipped {missing_path_count} rows."
        )
    if int(sample_index) > len(valid_rows):
        raise IndexError(
            f"--sample-index {sample_index} is outside the {len(valid_rows)} valid existing image rows in {manifest}."
        )

    manifest_row_index, row, resolved_path = valid_rows[int(sample_index) - 1]
    manifest_capture = _clean_text(row.get("capture"))
    inferred_capture = _infer_capture_from_filename(resolved_path.name)
    selected_capture = capture_cli or manifest_capture or inferred_capture or "plain"
    if capture_cli:
        capture_source = "cli"
    elif manifest_capture:
        capture_source = "manifest"
    elif inferred_capture:
        capture_source = "filename"
    else:
        capture_source = "default_plain"

    return SelectedImage(
        image_path=resolved_path,
        dataset=_clean_text(row.get("dataset")) or str(dataset),
        capture=selected_capture,
        capture_source=capture_source,
        source="manifest",
        manifest_path=manifest,
        manifest_row_index=manifest_row_index,
        valid_sample_index=int(sample_index),
        subject_id=_clean_text(row.get("subject_id")),
        frgp=_clean_text(row.get("frgp")),
        impression=_clean_text(row.get("impression")),
        ppi=_clean_text(row.get("ppi")),
        split=_clean_text(row.get("split")),
    )


def parse_methods(
    raw_methods: str | Sequence[str] | None,
    *,
    registry: Any,
) -> list[str]:
    if raw_methods is None:
        requested = list(DEFAULT_METHODS)
    elif isinstance(raw_methods, str):
        stripped = raw_methods.strip()
        if not stripped or stripped.lower() in {"default", "advisor", "advisor_default"}:
            requested = list(DEFAULT_METHODS)
        else:
            requested = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in raw_methods if str(item).strip()]
        if not requested:
            requested = list(DEFAULT_METHODS)

    methods: list[str] = []
    seen: set[str] = set()
    for method in requested:
        resolved = registry.resolve_retrieval_method(method)
        canonical = resolved.canonical_api_name
        if canonical in seen:
            continue
        methods.append(canonical)
        seen.add(canonical)
    return methods


def _coerce_vector(vector: Any) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def vector_sha256(vector: Any) -> str:
    arr = _coerce_vector(vector)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _float32_bits(vector: Any) -> np.ndarray:
    arr = _coerce_vector(vector)
    return arr.view(np.uint32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.array_equal(a, b):
        return 1.0
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    norm_a = float(np.linalg.norm(a64))
    norm_b = float(np.linalg.norm(b64))
    if norm_a <= 0.0 and norm_b <= 0.0:
        return 1.0
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    cosine = float(np.dot(a64, b64) / (norm_a * norm_b))
    if math.isfinite(cosine):
        return max(-1.0, min(1.0, cosine))
    return cosine


def compute_comparison_metrics(
    vector_run_1: Any,
    vector_run_2: Any,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    v1 = _coerce_vector(vector_run_1)
    v2 = _coerce_vector(vector_run_2)
    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes differ: {v1.shape} vs {v2.shape}")

    abs_diff = np.abs(v1 - v2)
    binary_metrics = compute_binary_comparison_metrics(v1, v2)
    return {
        "vector_dim": int(v1.size),
        "vector_1_sha256": vector_sha256(v1),
        "vector_2_sha256": vector_sha256(v2),
        "exact_equal": bool(np.array_equal(v1, v2)),
        "allclose_equal": bool(np.allclose(v1, v2, atol=float(epsilon), rtol=0.0)),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "nonzero_diff_count": int(np.count_nonzero(abs_diff)),
        "cosine_similarity": _cosine_similarity(v1, v2),
        **binary_metrics,
    }


def build_comparison_rows(vector_run_1: Any, vector_run_2: Any) -> list[dict[str, Any]]:
    v1 = _coerce_vector(vector_run_1)
    v2 = _coerce_vector(vector_run_2)
    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes differ: {v1.shape} vs {v2.shape}")
    abs_diff = np.abs(v1 - v2)
    return [
        {
            "dimension": int(index),
            "vector_run_1": float(v1[index]),
            "vector_run_2": float(v2[index]),
            "abs_diff": float(abs_diff[index]),
        }
        for index in range(v1.size)
    ]


def compute_binary_comparison_metrics(vector_run_1: Any, vector_run_2: Any) -> dict[str, Any]:
    bits_1 = _float32_bits(vector_run_1)
    bits_2 = _float32_bits(vector_run_2)
    if bits_1.shape != bits_2.shape:
        raise ValueError(f"Vector shapes differ: {bits_1.shape} vs {bits_2.shape}")

    total_dimensions = int(bits_1.size)
    total_binary_bit_count = int(total_dimensions * 32)
    binary_equal_dimension_count = int(np.count_nonzero(bits_1 == bits_2))
    differing_bit_count = sum(int(a ^ b).bit_count() for a, b in zip(bits_1, bits_2))
    binary_equal_bit_count = int(total_binary_bit_count - differing_bit_count)
    return {
        "binary_equal_dimension_count": binary_equal_dimension_count,
        "total_dimensions": total_dimensions,
        "binary_equal_dimension_rate": (
            float(binary_equal_dimension_count / total_dimensions)
            if total_dimensions
            else 1.0
        ),
        "binary_equal_bit_count": binary_equal_bit_count,
        "total_binary_bit_count": total_binary_bit_count,
        "binary_equal_bit_rate": (
            float(binary_equal_bit_count / total_binary_bit_count)
            if total_binary_bit_count
            else 1.0
        ),
    }


def build_binary_comparison_rows(vector_run_1: Any, vector_run_2: Any) -> list[dict[str, Any]]:
    v1 = _coerce_vector(vector_run_1)
    v2 = _coerce_vector(vector_run_2)
    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes differ: {v1.shape} vs {v2.shape}")

    bits_1 = _float32_bits(v1)
    bits_2 = _float32_bits(v2)
    abs_diff = np.abs(v1 - v2)
    return [
        {
            "dimension": int(index),
            "float_run_1": float(v1[index]),
            "float_run_2": float(v2[index]),
            "abs_diff": float(abs_diff[index]),
            "float32_hex_run_1": f"{int(bits_1[index]):08x}",
            "float32_hex_run_2": f"{int(bits_2[index]):08x}",
            "float32_binary_run_1": f"{int(bits_1[index]):032b}",
            "float32_binary_run_2": f"{int(bits_2[index]):032b}",
            "binary_equal": bool(bits_1[index] == bits_2[index]),
        }
        for index in range(v1.size)
    ]


def _format_float(value: Any, *, precision: int = 12) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{precision}g}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return _format_float(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def render_preview_markdown(
    *,
    method: str,
    comparison_rows: Sequence[Mapping[str, Any]],
    preview_dims: int,
    vector_dim: int,
) -> str:
    preview_count = min(max(0, int(preview_dims)), int(vector_dim))
    lines = [
        f"# Vector Comparison Preview: {method}",
        "",
        f"Showing the first {preview_count} of {vector_dim} zero-based vector dimensions.",
        "",
        "| dimension | vector_run_1 | vector_run_2 | abs_diff |",
        "|---:|---:|---:|---:|",
    ]
    for row in list(comparison_rows)[:preview_count]:
        lines.append(
            "| {dimension} | {v1} | {v2} | {diff} |".format(
                dimension=row.get("dimension", ""),
                v1=_format_float(row.get("vector_run_1"), precision=10),
                v2=_format_float(row.get("vector_run_2"), precision=10),
                diff=_format_float(row.get("abs_diff"), precision=10),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def render_summary_markdown(
    *,
    selected: SelectedImage,
    summary_rows: Sequence[Mapping[str, Any]],
    preview_markdowns: Mapping[str, str],
    binary_comparison_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    binary_csv_paths: Mapping[str, str],
    output_dir: Path,
    command: str,
    epsilon: float,
    preview_dims: int,
) -> str:
    lines: list[str] = []
    lines.append("# Vector Reproducibility Demo")
    lines.append("")
    lines.append(
        "Experiment goal: vectorize the same fingerprint image twice with the same "
        "retrieval method and configuration, then prove visually and numerically that "
        "the resulting retrieval vectors match."
    )
    lines.append("")
    lines.append(
        "This is a deterministic vectorization proof. It complements the 1:N self-match "
        "experiment by showing that the same image produces the same numeric vector "
        "under the same method configuration."
    )
    lines.append("")
    lines.append(
        "For floating-point DL models, allclose with epsilon is the scientific criterion "
        "if exact byte equality is not guaranteed on all hardware."
    )
    lines.append("")
    lines.append(
        "SHA-256 is included as a compact fingerprint of each full float32 vector. The "
        "binary comparison CSV files expose the actual IEEE-754 float32 bit pattern for "
        "every vector dimension, so the equality claim can be inspected dimension by "
        "dimension instead of trusting only a hash."
    )
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Epsilon: `{epsilon:g}`")
    lines.append(f"- Preview dimensions: `{preview_dims}`")
    lines.append("")
    lines.append("## Selected Image")
    lines.append("")
    lines.append(f"- Image path: `{selected.image_path}`")
    lines.append(f"- Dataset: `{selected.dataset}`")
    lines.append(f"- Capture: `{selected.capture}` ({selected.capture_source})")
    if selected.subject_id:
        lines.append(f"- Subject: `{selected.subject_id}`")
    if selected.frgp:
        lines.append(f"- FRGP: `{selected.frgp}`")
    if selected.impression:
        lines.append(f"- Impression: `{selected.impression}`")
    if selected.ppi:
        lines.append(f"- PPI: `{selected.ppi}`")
    if selected.manifest_path:
        lines.append(f"- Manifest: `{selected.manifest_path}`")
        lines.append(f"- Manifest row index: `{selected.manifest_row_index}`")
        lines.append(f"- Valid sample index: `{selected.valid_sample_index}`")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(
        "| method | vector_dim | exact_equal | allclose_equal | max_abs_diff | "
        "mean_abs_diff | nonzero_diff_count | cosine_similarity | "
        "binary_equal_dimension_count | total_dimensions | binary_equal_dimension_rate | "
        "binary_equal_bit_count | total_binary_bit_count | binary_equal_bit_rate | "
        "vector_1_sha256 | vector_2_sha256 |"
    )
    lines.append(
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )
    for row in summary_rows:
        lines.append(
            "| {method} | {dim} | {exact} | {allclose} | {max_diff} | {mean_diff} | "
            "{nonzero} | {cosine} | {binary_dims} | {total_dims} | {binary_dim_rate} | "
            "{binary_bits} | {total_bits} | {binary_bit_rate} | `{sha1}` | `{sha2}` |".format(
                method=row.get("method", ""),
                dim=row.get("vector_dim", ""),
                exact=_markdown_bool(row.get("exact_equal")),
                allclose=_markdown_bool(row.get("allclose_equal")),
                max_diff=_format_float(row.get("max_abs_diff")),
                mean_diff=_format_float(row.get("mean_abs_diff")),
                nonzero=row.get("nonzero_diff_count", ""),
                cosine=_format_float(row.get("cosine_similarity")),
                binary_dims=row.get("binary_equal_dimension_count", ""),
                total_dims=row.get("total_dimensions", ""),
                binary_dim_rate=_format_float(row.get("binary_equal_dimension_rate")),
                binary_bits=row.get("binary_equal_bit_count", ""),
                total_bits=row.get("total_binary_bit_count", ""),
                binary_bit_rate=_format_float(row.get("binary_equal_bit_rate")),
                sha1=row.get("vector_1_sha256", ""),
                sha2=row.get("vector_2_sha256", ""),
            )
        )
    lines.append("")
    lines.append("## Binary Comparison Files")
    lines.append("")
    lines.append(
        "The full binary comparison files are saved alongside the vector arrays and plots:"
    )
    lines.append("")
    for method in [str(row.get("method", "")) for row in summary_rows]:
        path = binary_csv_paths.get(method)
        if path:
            lines.append(f"- `{method}`: `{path}`")
    lines.append("")
    lines.append("## Vector Previews")
    lines.append("")
    for method in [str(row.get("method", "")) for row in summary_rows]:
        preview = preview_markdowns.get(method, "").strip()
        if not preview:
            continue
        preview_lines = preview.splitlines()
        if preview_lines and preview_lines[0].startswith("# "):
            preview_lines[0] = "### " + preview_lines[0][2:]
        lines.extend(preview_lines)
        lines.append("")
    lines.append("## Binary Previews")
    lines.append("")
    for method in [str(row.get("method", "")) for row in summary_rows]:
        rows = list(binary_comparison_rows.get(method, ()))[:20]
        if not rows:
            continue
        lines.append(f"### Float32 Binary Preview: {method}")
        lines.append("")
        lines.append("Showing the first 20 zero-based vector dimensions.")
        lines.append("")
        lines.append(
            "| dimension | float_run_1 | float_run_2 | float32_hex_run_1 | "
            "float32_hex_run_2 | binary_equal |"
        )
        lines.append("|---:|---:|---:|---|---|---|")
        for row in rows:
            lines.append(
                "| {dimension} | {float_1} | {float_2} | `{hex_1}` | `{hex_2}` | {equal} |".format(
                    dimension=row.get("dimension", ""),
                    float_1=_format_float(row.get("float_run_1"), precision=10),
                    float_2=_format_float(row.get("float_run_2"), precision=10),
                    hex_1=row.get("float32_hex_run_1", ""),
                    hex_2=row.get("float32_hex_run_2", ""),
                    equal=_markdown_bool(row.get("binary_equal")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def write_overlay_plot(path: Path, *, method: str, vector_run_1: Any, vector_run_2: Any) -> None:
    plt = _load_pyplot()
    v1 = _coerce_vector(vector_run_1)
    v2 = _coerce_vector(vector_run_2)
    x = np.arange(v1.size)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, v1, label="vector_run_1", linewidth=1.0)
    ax.plot(x, v2, label="vector_run_2", linewidth=1.0, linestyle="--")
    ax.set_title(f"{method} retrieval vector overlay")
    ax.set_xlabel("dimension")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_diff_plot(path: Path, *, method: str, vector_run_1: Any, vector_run_2: Any) -> None:
    plt = _load_pyplot()
    v1 = _coerce_vector(vector_run_1)
    v2 = _coerce_vector(vector_run_2)
    abs_diff = np.abs(v1 - v2)
    x = np.arange(abs_diff.size)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, abs_diff, linewidth=1.0)
    ax.set_title(f"{method} absolute vector difference")
    ax.set_xlabel("dimension")
    ax.set_ylabel("abs_diff")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_runtime_retrieval_vector(
    *,
    match_service: Any,
    method: str,
    image_path: Path,
    capture: str,
) -> np.ndarray:
    from apps.api.identification_service import (  # noqa: WPS433 - imported lazily for pure helper tests.
        _MATCH_SERVICE_EMBED_METHODS,
        _MATCH_SERVICE_MODEL_ATTRS,
        _l2_normalize,
    )

    match_service.ensure_method_available(method)

    embed_method_name = _MATCH_SERVICE_EMBED_METHODS.get(method)
    if embed_method_name is not None:
        embed_method = getattr(match_service, embed_method_name, None)
        if not callable(embed_method):
            raise RuntimeError(f"No MatchService vectorizer adapter is registered for {method!r}")
        raw_vector = embed_method(str(image_path), capture=capture)
    else:
        model_attr = _MATCH_SERVICE_MODEL_ATTRS.get(method)
        if model_attr is None:
            raise RuntimeError(f"No MatchService vectorizer adapter is registered for {method!r}")
        model = getattr(match_service, model_attr, None)
        if model is None:
            raise RuntimeError(f"MatchService has no loaded model for retrieval method {method!r}")
        raw_vector = model.embed_path(str(image_path), capture=capture)[0]

    return _coerce_vector(_l2_normalize(np.asarray(raw_vector, dtype=np.float32)))


def _format_command(argv: Sequence[str]) -> str:
    parts = [sys.executable, *argv]
    try:
        return subprocess.list2cmdline([str(part) for part in parts])
    except Exception:
        return " ".join(str(part) for part in parts)


def _run_git(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_info() -> dict[str, Any]:
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]) or "unknown",
        "dirty": bool(status),
        "status_porcelain": status or "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Vectorize the same fingerprint image twice with each retrieval method and "
            "write advisor-facing numeric and visual reproducibility evidence."
        )
    )
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path. Defaults to data/manifests/<dataset>/manifest.csv.",
    )
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods. Defaults to {','.join(DEFAULT_METHODS)}.",
    )
    parser.add_argument("--capture", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dims", type=int, default=DEFAULT_PREVIEW_DIMS)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser


def run_demo(args: argparse.Namespace) -> int:
    from apps.api.method_registry import load_api_method_registry  # noqa: WPS433
    from apps.api.service import MatchService  # noqa: WPS433

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    command = _format_command(sys.argv)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_api_method_registry()
    methods = parse_methods(args.methods, registry=registry)
    preview_dims = max(0, int(args.preview_dims))
    epsilon = float(args.epsilon)
    selected = select_image(
        image_path=args.image_path,
        dataset=str(args.dataset),
        manifest_path=args.manifest,
        sample_index=int(args.sample_index),
        capture=args.capture,
    )

    print("=== Vector reproducibility demo ===")
    print(f"Image      : {selected.image_path}")
    print(f"Dataset    : {selected.dataset}")
    print(f"Capture    : {selected.capture} ({selected.capture_source})")
    print(f"Methods    : {','.join(methods)}")
    print(f"Output dir : {output_dir}")

    match_service = MatchService()
    summary_rows: list[dict[str, Any]] = []
    preview_markdowns: dict[str, str] = {}
    binary_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    binary_csv_paths: dict[str, str] = {}
    method_outputs: dict[str, dict[str, str]] = {}

    for method in methods:
        print(f"[vectorize] {method} run 1")
        vector_run_1 = compute_runtime_retrieval_vector(
            match_service=match_service,
            method=method,
            image_path=selected.image_path,
            capture=selected.capture,
        )
        print(f"[vectorize] {method} run 2")
        vector_run_2 = compute_runtime_retrieval_vector(
            match_service=match_service,
            method=method,
            image_path=selected.image_path,
            capture=selected.capture,
        )

        metrics = compute_comparison_metrics(vector_run_1, vector_run_2, epsilon=epsilon)
        comparison_rows = build_comparison_rows(vector_run_1, vector_run_2)
        binary_comparison_rows = build_binary_comparison_rows(vector_run_1, vector_run_2)

        vector_1_path = output_dir / f"vector_{method}_run1.npy"
        vector_2_path = output_dir / f"vector_{method}_run2.npy"
        comparison_csv_path = output_dir / f"vector_{method}_comparison.csv"
        binary_comparison_csv_path = output_dir / f"vector_{method}_binary_comparison.csv"
        preview_md_path = output_dir / f"vector_{method}_comparison_preview.md"
        overlay_path = output_dir / f"vector_{method}_overlay.png"
        diff_path = output_dir / f"vector_{method}_diff.png"

        np.save(vector_1_path, vector_run_1)
        np.save(vector_2_path, vector_run_2)
        write_csv(comparison_csv_path, comparison_rows, COMPARISON_COLUMNS)
        write_csv(binary_comparison_csv_path, binary_comparison_rows, BINARY_COMPARISON_COLUMNS)

        preview_md = render_preview_markdown(
            method=method,
            comparison_rows=comparison_rows,
            preview_dims=preview_dims,
            vector_dim=int(metrics["vector_dim"]),
        )
        preview_md_path.write_text(preview_md, encoding="utf-8")
        preview_markdowns[method] = preview_md

        write_overlay_plot(
            overlay_path,
            method=method,
            vector_run_1=vector_run_1,
            vector_run_2=vector_run_2,
        )
        write_diff_plot(
            diff_path,
            method=method,
            vector_run_1=vector_run_1,
            vector_run_2=vector_run_2,
        )

        row = {"method": method, **metrics}
        summary_rows.append(row)
        binary_rows_by_method[method] = binary_comparison_rows
        binary_csv_paths[method] = str(binary_comparison_csv_path)
        method_outputs[method] = {
            "vector_run_1": str(vector_1_path),
            "vector_run_2": str(vector_2_path),
            "comparison_csv": str(comparison_csv_path),
            "binary_comparison_csv": str(binary_comparison_csv_path),
            "comparison_preview_md": str(preview_md_path),
            "overlay_png": str(overlay_path),
            "diff_png": str(diff_path),
        }
        print(
            "[compare] {method}: exact={exact} allclose={allclose} max_abs_diff={max_diff}".format(
                method=method,
                exact=metrics["exact_equal"],
                allclose=metrics["allclose_equal"],
                max_diff=_format_float(metrics["max_abs_diff"]),
            )
        )

    summary_csv_path = output_dir / "vector_reproducibility_summary.csv"
    summary_md_path = output_dir / "vector_reproducibility_summary.md"
    run_manifest_path = output_dir / "run_manifest.json"

    write_csv(summary_csv_path, summary_rows, SUMMARY_COLUMNS)
    summary_md = render_summary_markdown(
        selected=selected,
        summary_rows=summary_rows,
        preview_markdowns=preview_markdowns,
        binary_comparison_rows=binary_rows_by_method,
        binary_csv_paths=binary_csv_paths,
        output_dir=output_dir,
        command=command,
        epsilon=epsilon,
        preview_dims=preview_dims,
    )
    summary_md_path.write_text(summary_md, encoding="utf-8")

    run_manifest = {
        "run_timestamp": timestamp,
        "command": command,
        "repo_root": str(REPO_ROOT),
        "selected_image": selected.to_json(),
        "methods": methods,
        "requested_methods": args.methods,
        "epsilon": epsilon,
        "preview_dims": preview_dims,
        "outputs": {
            "run_manifest": str(run_manifest_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "methods": method_outputs,
        },
        "summary": summary_rows,
        "git": collect_git_info(),
    }
    write_json(run_manifest_path, run_manifest)

    print(f"[done] wrote {summary_md_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
