from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


BACKUP_SUFFIXES = {".csv", ".json", ".md", ".txt"}
MISSING_VALUES = {"", "nan", "none", "null"}
PATH_FIELD_NAMES = {
    "csv",
    "json",
    "meta_json",
    "method_meta_json",
    "out_dir",
    "outdir",
    "output",
    "output_dir",
    "roc_png",
    "scores_csv",
    "source_dir",
    "summary_csv",
}
STALE_REGEN_MARKER = "_regen_h5_tmp_"


@dataclass
class RepairReport:
    outdir: Path
    changed_files: list[Path] = field(default_factory=list)
    backups: list[Path] = field(default_factory=list)
    repaired_rows: int = 0
    repaired_run_meta: int = 0
    repaired_manifest: bool = False


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_VALUES


def backup_path_for(path: Path, timestamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.repair_benchmark_metadata_{timestamp}.bak")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.repair_benchmark_metadata_{timestamp}_{index}.bak")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a backup path for {path}")


def write_if_changed(path: Path, data: str, *, timestamp: str, report: RepairReport) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == data:
        return False

    if path.exists() and path.suffix.lower() in BACKUP_SUFFIXES:
        backup = backup_path_for(path, timestamp)
        shutil.copy2(path, backup)
        report.backups.append(backup)

    path.write_text(data, encoding="utf-8", newline="")
    report.changed_files.append(path)
    return True


def json_dumps_config(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def json_dumps_file(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def read_summary_rows(summary_csv: Path) -> tuple[list[str], list[dict[str, str]], str]:
    raw = summary_csv.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None:
        raise ValueError(f"{summary_csv} is empty or missing a header")
    return list(reader.fieldnames), [dict(row) for row in reader], newline


def write_summary_rows(fieldnames: list[str], rows: Iterable[Mapping[str, Any]], *, newline: str) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=newline)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return handle.getvalue()


def update_config_json(raw: str, outdir: Path) -> str:
    config = json.loads(str(raw))
    if isinstance(config.get("fusion"), dict):
        config["fusion"]["source_dir"] = str(outdir)
    return json_dumps_config(config)


def repair_stale_bundle_path(value: str, outdir: Path) -> str:
    if STALE_REGEN_MARKER not in value:
        return value

    normalized = value.replace("\\", "/")
    marker_index = normalized.find(STALE_REGEN_MARKER)
    if marker_index < 0:
        return value

    run_token = f"/{outdir.name}"
    run_index = normalized.find(run_token, marker_index)
    if run_index < 0:
        return value

    suffix = normalized[run_index + len(run_token):]
    suffix_parts = [part for part in suffix.split("/") if part]
    if suffix_parts:
        return str(outdir.joinpath(*suffix_parts))
    return str(outdir)


def should_treat_as_path_field(key: str) -> bool:
    lowered = key.lower()
    return lowered in PATH_FIELD_NAMES or lowered.endswith("_path") or lowered.endswith("_csv") or lowered.endswith("_json")


def repair_plain_path_fields(payload: Any, outdir: Path, *, parent_key: str = "") -> Any:
    if isinstance(payload, dict):
        repaired: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "config_json":
                repaired[key] = value
                continue
            repaired[key] = repair_plain_path_fields(value, outdir, parent_key=key)
        return repaired

    if isinstance(payload, list):
        return [repair_plain_path_fields(item, outdir, parent_key=parent_key) for item in payload]

    if isinstance(payload, str):
        if should_treat_as_path_field(parent_key) or STALE_REGEN_MARKER in payload:
            return repair_stale_bundle_path(payload, outdir)
    return payload


def repaired_summary_rows(outdir: Path) -> tuple[list[str], list[dict[str, str]], str]:
    summary_csv = outdir / "results_summary.csv"
    fieldnames, rows, newline = read_summary_rows(summary_csv)
    required = {"method", "split", "scores_csv", "config_json"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{summary_csv} missing required columns: {missing}")

    for row in rows:
        method = row["method"]
        split = row["split"]
        row["scores_csv"] = str(outdir / f"scores_{method}_{split}.csv")
        if "meta_json" in row and not is_missing(row.get("meta_json")):
            row["meta_json"] = str(outdir / Path(str(row["meta_json"])).name)
        if not is_missing(row.get("config_json")):
            row["config_json"] = update_config_json(str(row["config_json"]), outdir)

    return fieldnames, rows, newline


def row_lookup(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("method", "")), str(row.get("split", "")))
        if key in lookup:
            raise ValueError(f"Duplicate summary row for method/split: {key}")
        lookup[key] = dict(row)
    return lookup


def sync_run_meta(path: Path, outdir: Path, summary_row: Mapping[str, str], *, timestamp: str, report: RepairReport) -> None:
    method = str(summary_row["method"])
    split = str(summary_row["split"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = repair_plain_path_fields(payload, outdir)

    payload["row"] = dict(summary_row)
    payload["scores_csv"] = str(outdir / f"scores_{method}_{split}.csv")
    payload["roc_png"] = str(outdir / f"roc_{method}_{split}.png")
    payload["summary_csv"] = str(outdir / "results_summary.csv")

    meta_json = summary_row.get("meta_json")
    payload["method_meta_json"] = None if is_missing(meta_json) else str(meta_json)

    config = json.loads(str(summary_row["config_json"]))
    payload["config"] = config
    for field_name in ("resolved_data_dir", "manifest_path", "pairs_path"):
        if field_name in config:
            payload[field_name] = config[field_name]

    if write_if_changed(path, json_dumps_file(payload), timestamp=timestamp, report=report):
        report.repaired_run_meta += 1


def sync_run_manifest(path: Path, outdir: Path, *, timestamp: str, report: RepairReport) -> None:
    if not path.exists():
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = repair_plain_path_fields(payload, outdir)
    if isinstance(payload, dict):
        for key in ("outdir", "out_dir", "output", "output_dir", "source_dir"):
            if key in payload and not is_missing(payload[key]):
                payload[key] = str(outdir)

    if write_if_changed(path, json_dumps_file(payload), timestamp=timestamp, report=report):
        report.repaired_manifest = True


def repair_bundle_metadata(outdir: str | Path) -> RepairReport:
    resolved_outdir = resolve_path(outdir)
    if not resolved_outdir.exists():
        raise FileNotFoundError(f"Bundle outdir not found: {resolved_outdir}")

    report = RepairReport(outdir=resolved_outdir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_csv = resolved_outdir / "results_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    fieldnames, rows, newline = repaired_summary_rows(resolved_outdir)
    summary_text = write_summary_rows(fieldnames, rows, newline=newline)
    if write_if_changed(summary_csv, summary_text, timestamp=timestamp, report=report):
        report.repaired_rows = len(rows)

    by_key = row_lookup(rows)
    for row in rows:
        method = row["method"]
        split = row["split"]
        run_meta = resolved_outdir / f"run_{method}_{split}.meta.json"
        if not run_meta.exists():
            raise FileNotFoundError(f"Missing run meta for {method}/{split}: {run_meta}")
        sync_run_meta(run_meta, resolved_outdir, by_key[(method, split)], timestamp=timestamp, report=report)

    sync_run_manifest(resolved_outdir / "run_manifest.json", resolved_outdir, timestamp=timestamp, report=report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair promoted benchmark bundle metadata paths.")
    parser.add_argument("--outdir", required=True, help="Benchmark bundle directory to repair.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = repair_bundle_metadata(args.outdir)
    print(f"[OK] Repaired benchmark metadata: {report.outdir}")
    print(f"[OK] Changed files: {len(report.changed_files)}")
    for path in report.changed_files:
        print(f"  - {path}")
    print(f"[OK] Backups: {len(report.backups)}")
    for path in report.backups:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
