from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from PIL import Image

SOURCE_DATASET = "polyu_cross"

MANIFEST_COLUMNS = [
    "sample_id",
    "subject_id",
    "finger_id",
    "session_id",
    "modality",
    "capture_id",
    "image_path",
    "width",
    "height",
    "source_dataset",
]

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_MODALITIES = {"contactless_2d", "contact_based_2d"}
ALLOWED_SESSIONS = {"session_1", "session_2", "unknown"}


class PolyUCrossManifestError(RuntimeError):
    """Raised when the PolyU Cross manifest cannot be built safely."""


@dataclass(frozen=True)
class ManifestBuildResult:
    manifest_path: Path
    warnings_path: Path
    sanity_report_path: Path
    row_count: int
    warning_count: int


@dataclass(frozen=True)
class ParsedIds:
    subject_id: str
    finger_id: str
    capture_id: str
    layout: str


def iter_image_files(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"PolyU Cross dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"PolyU Cross dataset root is not a directory: {dataset_root}")
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_manifest_dataframe(
    dataset_root: Path,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    dataset_root = dataset_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for image_path in iter_image_files(dataset_root):
        relative_path = _relative_path(image_path, dataset_root)
        modality = infer_modality(image_path, dataset_root)
        if modality is None:
            warnings.append(
                _warning(
                    relative_path,
                    "unknown_modality",
                    "Could not infer contactless/contact-based modality from path.",
                )
            )
            continue

        parsed_ids = parse_ids(image_path, dataset_root, modality=modality)
        if parsed_ids is None:
            warnings.append(
                _warning(
                    relative_path,
                    "unparseable_metadata",
                    "Could not parse subject_id, finger_id, and capture_id from filename/folders.",
                )
            )
            continue

        dimensions = _read_dimensions(image_path)
        if dimensions is None:
            warnings.append(
                _warning(
                    relative_path,
                    "unreadable_image",
                    "Could not read image dimensions with Pillow.",
                )
            )
            continue
        width, height = dimensions

        session_id = infer_session_id(image_path, dataset_root)
        row = {
            "subject_id": parsed_ids.subject_id,
            "finger_id": parsed_ids.finger_id,
            "session_id": session_id,
            "modality": modality,
            "capture_id": parsed_ids.capture_id,
            "image_path": relative_path,
            "width": int(width),
            "height": int(height),
            "source_dataset": SOURCE_DATASET,
        }
        row["sample_id"] = make_sample_id(row)
        rows.append(row)

    if strict and warnings:
        details = "; ".join(
            f"{entry['reason']}:{entry['image_path']}" for entry in warnings[:5]
        )
        raise PolyUCrossManifestError(
            f"Strict PolyU Cross manifest build found {len(warnings)} unparseable image file(s): {details}"
        )

    if not rows:
        raise PolyUCrossManifestError(
            f"No parseable PolyU Cross image files found under {dataset_root}"
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = df.sort_values(
        ["modality", "subject_id", "finger_id", "session_id", "capture_id", "image_path"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_manifest_dataframe(df)
    return df, warnings


def write_manifest(
    dataset_root: Path,
    output: Path,
    *,
    strict: bool = False,
) -> ManifestBuildResult:
    output = output.expanduser()
    df, warnings = build_manifest_dataframe(dataset_root, strict=strict)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    warnings_path = output.parent / "warnings.json"
    warnings_path.write_text(json.dumps(warnings, indent=2), encoding="utf-8")

    sanity_report = build_sanity_report(df, warnings, output=output, warnings_path=warnings_path)
    sanity_report_path = output.parent / "sanity_report.json"
    sanity_report_path.write_text(json.dumps(sanity_report, indent=2), encoding="utf-8")

    return ManifestBuildResult(
        manifest_path=output,
        warnings_path=warnings_path,
        sanity_report_path=sanity_report_path,
        row_count=int(len(df)),
        warning_count=int(len(warnings)),
    )


def parse_ids(path: Path, dataset_root: Path, *, modality: str) -> Optional[ParsedIds]:
    rel = path.relative_to(dataset_root)
    text = "/".join(part.lower() for part in rel.with_suffix("").parts)

    named_subject = _find_named_value(
        text,
        [
            "subject_id",
            "subject",
            "subj",
            "sub",
            "client_id",
            "client",
            "user_id",
            "user",
            "uid",
            "identity",
            "person",
        ],
    )
    named_finger = _find_named_value(
        text,
        ["finger_id", "finger", "fgr", "frgp", "fid", "digit"],
    )
    named_capture = _find_named_value(
        text,
        [
            "capture_id",
            "capture",
            "cap",
            "sample_id",
            "sample",
            "impression",
            "img",
            "image",
            "shot",
        ],
    )

    stem_numbers = [_normalize_numeric_id(value) for value in re.findall(r"\d+", path.stem)]
    parent_p = _last_parent_p_value(rel)
    stem_p = _p_value(path.stem)
    two_number_stem = re.fullmatch(r"0*(\d+)[_-]0*(\d+)", path.stem)

    subject_id = named_subject
    finger_id = named_finger
    capture_id = named_capture
    layout = "named_fields"

    if subject_id and finger_id and capture_id:
        return ParsedIds(subject_id, finger_id, capture_id, layout)

    if len(stem_numbers) >= 3:
        subject_id = subject_id or stem_numbers[0]
        finger_id = finger_id or stem_numbers[1]
        capture_id = capture_id or stem_numbers[2]
        layout = "three_number_stem"
    elif parent_p is not None and stem_p is not None:
        # The PolyU readme defines pX/pY as user finger ID X and image sample Y.
        subject_id = subject_id or parent_p
        finger_id = finger_id or parent_p
        capture_id = capture_id or stem_p
        layout = "polyu_px_py"
    elif modality == "contact_based_2d" and two_number_stem is not None:
        # The PolyU readme defines X_Y as user finger ID X and image sample Y.
        identity = _normalize_numeric_id(two_number_stem.group(1))
        subject_id = subject_id or identity
        finger_id = finger_id or identity
        capture_id = capture_id or _normalize_numeric_id(two_number_stem.group(2))
        layout = "polyu_x_y"
    elif subject_id and finger_id and stem_numbers:
        capture_id = capture_id or stem_numbers[-1]
        layout = "named_subject_finger_stem_capture"
    elif subject_id and len(stem_numbers) >= 2:
        finger_id = finger_id or stem_numbers[-2]
        capture_id = capture_id or stem_numbers[-1]
        layout = "named_subject_two_number_stem"
    elif finger_id and len(stem_numbers) >= 2:
        subject_id = subject_id or stem_numbers[0]
        capture_id = capture_id or stem_numbers[-1]
        layout = "named_finger_two_number_stem"
    elif len(stem_numbers) == 2:
        identity = stem_numbers[0]
        subject_id = subject_id or identity
        finger_id = finger_id or identity
        capture_id = capture_id or stem_numbers[1]
        layout = "two_number_stem_identity_capture"
    elif parent_p is not None and stem_numbers:
        subject_id = subject_id or parent_p
        finger_id = finger_id or parent_p
        capture_id = capture_id or stem_numbers[-1]
        layout = "p_parent_numeric_stem_capture"

    if not (subject_id and finger_id and capture_id):
        return None
    return ParsedIds(subject_id, finger_id, capture_id, layout)


def infer_modality(path: Path, dataset_root: Path) -> Optional[str]:
    parts = [_slug(part) for part in path.relative_to(dataset_root).parts]
    if any("contact_based" in part or "contact_base" in part for part in parts):
        return "contact_based_2d"
    if any("contactless" in part for part in parts):
        return "contactless_2d"
    return None


def infer_session_id(path: Path, dataset_root: Path) -> str:
    for part in path.relative_to(dataset_root).parts:
        slug = _slug(part)
        if re.search(r"(^|_)first($|_)", slug) or re.search(r"(^|_)1st($|_)", slug):
            return "session_1"
        if re.search(r"(^|_)second($|_)", slug) or re.search(r"(^|_)2nd($|_)", slug):
            return "session_2"

        for pattern in (
            r"(?:^|_)(?:session|sess|visit)(?:_|)(\d+)(?:$|_)",
            r"(?:^|_)(\d+)(?:st|nd|rd|th)?_(?:session|sess|visit)(?:$|_)",
            r"^s(\d+)$",
        ):
            match = re.search(pattern, slug)
            if not match:
                continue
            value = int(match.group(1))
            if value == 1:
                return "session_1"
            if value == 2:
                return "session_2"
    return "unknown"


def make_sample_id(row: dict[str, object]) -> str:
    key = "|".join(
        str(row[column])
        for column in ["modality", "subject_id", "finger_id", "session_id", "capture_id", "image_path"]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{SOURCE_DATASET}_{digest}"


def build_sanity_report(
    df: pd.DataFrame,
    warnings: list[dict[str, object]],
    *,
    output: Path,
    warnings_path: Path,
) -> dict[str, object]:
    checks = {
        "columns_exact": list(df.columns) == MANIFEST_COLUMNS,
        "sample_id_unique": bool(df["sample_id"].is_unique),
        "allowed_modalities": set(df["modality"].dropna().unique()).issubset(ALLOWED_MODALITIES),
        "allowed_sessions": set(df["session_id"].dropna().unique()).issubset(ALLOWED_SESSIONS),
        "source_dataset_constant": set(df["source_dataset"].dropna().unique()) == {SOURCE_DATASET},
        "image_paths_relative": all(not _looks_absolute(str(value)) for value in df["image_path"].tolist()),
    }
    warning_counts: dict[str, int] = {}
    for item in warnings:
        reason = str(item.get("reason", "unknown"))
        warning_counts[reason] = warning_counts.get(reason, 0) + 1

    return {
        "schema_version": "polyu_cross_manifest_sanity_v1",
        "source_dataset": SOURCE_DATASET,
        "manifest_file": output.name,
        "warnings_file": warnings_path.name,
        "manifest": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "unique_sample_ids": int(df["sample_id"].nunique()),
            "unique_subject_ids": int(df["subject_id"].nunique()),
            "unique_finger_ids": int(df["finger_id"].nunique()),
            "modality_counts": {str(k): int(v) for k, v in df["modality"].value_counts().to_dict().items()},
            "session_counts": {str(k): int(v) for k, v in df["session_id"].value_counts().to_dict().items()},
            "rows_missing_dimensions": int(df[["width", "height"]].isna().any(axis=1).sum()),
        },
        "warnings": {
            "count": int(len(warnings)),
            "by_reason": warning_counts,
        },
        "checks": checks,
        "ok": bool(all(checks.values())),
        "notes": [
            "The PolyU readme exposes a user finger identification number, not a separate person/finger decomposition, for pX/pY and X_Y layouts.",
            "For those canonical layouts, subject_id and finger_id both use the parsed user finger identification number.",
            "No biometric images are copied or modified by this manifest builder.",
        ],
    }


def _validate_manifest_dataframe(df: pd.DataFrame) -> None:
    if list(df.columns) != MANIFEST_COLUMNS:
        raise PolyUCrossManifestError(f"Manifest columns do not match required schema: {list(df.columns)}")
    bad_modalities = sorted(set(df["modality"].dropna().unique()) - ALLOWED_MODALITIES)
    if bad_modalities:
        raise PolyUCrossManifestError(f"Unsupported PolyU Cross modality values: {bad_modalities}")
    bad_sessions = sorted(set(df["session_id"].dropna().unique()) - ALLOWED_SESSIONS)
    if bad_sessions:
        raise PolyUCrossManifestError(f"Unsupported PolyU Cross session_id values: {bad_sessions}")
    if set(df["source_dataset"].dropna().unique()) != {SOURCE_DATASET}:
        raise PolyUCrossManifestError("source_dataset must be constant 'polyu_cross'")
    if not df["sample_id"].is_unique:
        duplicates = df[df["sample_id"].duplicated()]["sample_id"].head(5).tolist()
        raise PolyUCrossManifestError(f"sample_id values are not unique. Examples: {duplicates}")


def _read_dimensions(path: Path) -> Optional[tuple[int, int]]:
    try:
        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        return None


def _find_named_value(text: str, names: list[str]) -> Optional[str]:
    for name in names:
        pattern = rf"(?:^|[^a-z0-9]){re.escape(name)}[^a-z0-9]*(\d+)(?:$|[^a-z0-9])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_numeric_id(match.group(1))
    return None


def _last_parent_p_value(relative_path: Path) -> Optional[str]:
    for part in reversed(relative_path.parts[:-1]):
        value = _p_value(part)
        if value is not None:
            return value
    return None


def _p_value(text: str) -> Optional[str]:
    match = re.fullmatch(r"p0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return _normalize_numeric_id(match.group(1))
    return None


def _normalize_numeric_id(value: str) -> str:
    value = str(value)
    normalized = value.lstrip("0")
    return normalized or "0"


def _relative_path(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _warning(image_path: str, reason: str, message: str) -> dict[str, object]:
    return {
        "image_path": image_path,
        "reason": reason,
        "message": message,
    }


def _looks_absolute(path: str) -> bool:
    return Path(path).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", path))
