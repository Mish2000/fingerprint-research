from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_professor_1000_pos_neg"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
PAIR_SETS = ("positive_1000", "negative_1000")
METHODS = ("sift", "minutiae")
PATH_RE = re.compile(
    r"(?P<subject>\d+)[_-](?P<capture>plain|roll|rolled)[_-](?P<set>\d+)[_-](?P<frgp>\d+)\.[^.\\/]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedPath:
    subject: str
    capture: str
    frgp: str


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _norm_path(value: Any) -> str:
    return str(value).strip().replace("/", "\\")


def _parse_path(value: Any) -> ParsedPath | None:
    text = _norm_path(value)
    match = PATH_RE.search(text)
    if not match:
        return None
    capture = match.group("capture").lower()
    if capture == "rolled":
        capture = "roll"
    return ParsedPath(
        subject=str(int(match.group("subject"))),
        capture=capture,
        frgp=str(int(match.group("frgp"))),
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _issue(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    row_index: int | None,
    pair_set: str,
    kind: str,
    status: str,
    path_a: Any = "",
    path_b: Any = "",
    details: str = "",
) -> None:
    rows.append(
        {
            "source_file": str(source_file),
            "row_index": "" if row_index is None else int(row_index),
            "pair_set": pair_set,
            "kind": kind,
            "status": status,
            "path_a": _norm_path(path_a) if path_a != "" else "",
            "path_b": _norm_path(path_b) if path_b != "" else "",
            "details": details,
        }
    )


def _expected_label(pair_set: str) -> int:
    return 1 if pair_set == "positive_1000" else 0


def _validate_pair_rows(path: Path, pair_set: str, df: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    required = {"path_a", "path_b", "label"}
    missing = required - set(df.columns)
    if missing:
        _issue(
            rows,
            source_file=path,
            row_index=None,
            pair_set=pair_set,
            kind="required_columns",
            status="fail",
            details=f"missing columns: {sorted(missing)}",
        )
        return

    seen: dict[tuple[str, str], int] = {}
    expected_label = _expected_label(pair_set)
    for idx, row in df.iterrows():
        row_num = int(idx) + 2
        path_a = row["path_a"]
        path_b = row["path_b"]
        pair_key = (_norm_path(path_a).lower(), _norm_path(path_b).lower())
        if pair_key in seen:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="duplicate_pair_paths",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"duplicates data row {seen[pair_key]}",
            )
        else:
            seen[pair_key] = row_num

        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            label = -1
        if label != expected_label:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="label_value",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"label={row['label']!r}, expected {expected_label}",
            )

        parsed_a = _parse_path(path_a)
        parsed_b = _parse_path(path_b)
        if parsed_a is None or parsed_b is None:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="path_parse",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"parse_a={parsed_a is not None}; parse_b={parsed_b is not None}",
            )
            continue

        if parsed_a.capture != "plain" or parsed_b.capture != "roll":
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="capture_order",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"path_a={parsed_a.capture}, path_b={parsed_b.capture}",
            )
        if parsed_a.frgp != parsed_b.frgp:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="frgp_consistency",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"path_a frgp={parsed_a.frgp}, path_b frgp={parsed_b.frgp}",
            )
        same_subject = parsed_a.subject == parsed_b.subject
        if pair_set == "positive_1000" and not same_subject:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="subject_consistency",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"path_a subject={parsed_a.subject}, path_b subject={parsed_b.subject}",
            )
        if pair_set == "negative_1000" and same_subject:
            _issue(
                rows,
                source_file=path,
                row_index=row_num,
                pair_set=pair_set,
                kind="subject_consistency",
                status="fail",
                path_a=path_a,
                path_b=path_b,
                details=f"negative pair has same subject={parsed_a.subject}",
            )


def _validate_alignment(
    *,
    selected_path: Path,
    selected: pd.DataFrame | None,
    score_path: Path,
    scores: pd.DataFrame | None,
    pair_set: str,
    rows: list[dict[str, Any]],
) -> None:
    if scores is None:
        _issue(
            rows,
            source_file=score_path,
            row_index=None,
            pair_set=pair_set,
            kind="file_exists",
            status="fail",
            details="score CSV missing",
        )
        return
    _validate_pair_rows(score_path, pair_set, scores, rows)
    if selected is None:
        _issue(
            rows,
            source_file=selected_path,
            row_index=None,
            pair_set=pair_set,
            kind="file_exists",
            status="fail",
            details="selected pair CSV missing; score alignment skipped",
        )
        return

    if len(selected) != len(scores):
        _issue(
            rows,
            source_file=score_path,
            row_index=None,
            pair_set=pair_set,
            kind="selected_score_alignment",
            status="fail",
            details=f"row-count mismatch selected={len(selected)} scores={len(scores)}",
        )
        return

    for idx, (selected_row, score_row) in enumerate(zip(selected.to_dict("records"), scores.to_dict("records"))):
        row_num = int(idx) + 2
        fields = ["path_a", "path_b", "label"]
        if "split" in selected.columns and "split" in scores.columns:
            fields.append("split")
        mismatches = []
        for field in fields:
            left = str(selected_row.get(field, "")).strip()
            right = str(score_row.get(field, "")).strip()
            if field.startswith("path_"):
                left = _norm_path(left).lower()
                right = _norm_path(right).lower()
            if left != right:
                mismatches.append(f"{field}: selected={selected_row.get(field)!r}, score={score_row.get(field)!r}")
        if mismatches:
            _issue(
                rows,
                source_file=score_path,
                row_index=row_num,
                pair_set=pair_set,
                kind="selected_score_alignment",
                status="fail",
                path_a=score_row.get("path_a", ""),
                path_b=score_row.get("path_b", ""),
                details="; ".join(mismatches),
            )


def _summary_counts(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame(columns=["kind", "status", "count"])
    return (
        report.groupby(["kind", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status", "kind"])
    )


def _render_markdown(*, benchmark_dir: Path, report: pd.DataFrame, checked_files: list[Path]) -> str:
    failures = report[report["status"] == "fail"].copy() if not report.empty else report
    summary = _summary_counts(report)
    lines = [
        "# Professor 1000 Pair Label/Path Validation",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        "",
        "## Files Checked",
        "",
    ]
    for path in checked_files:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Summary", ""])
    if failures.empty:
        lines.append("- PASS: no label/path, duplicate, parse, or selected-score alignment failures were found.")
    else:
        lines.append(f"- FAIL: {len(failures)} validation failures were found.")
    lines.append("")
    lines.extend(["| kind | status | count |", "| --- | --- | ---: |"])
    if summary.empty:
        lines.append("| all | ok | 0 |")
    else:
        for _, row in summary.iterrows():
            lines.append(f"| {row['kind']} | {row['status']} | {int(row['count'])} |")

    lines.extend(["", "## Findings", ""])
    if failures.empty:
        lines.append("- Positive pairs parse as same-subject plain-vs-roll rows with matching FRGP.")
        lines.append("- Negative pairs parse as different-subject plain-vs-roll rows with matching FRGP.")
        lines.append("- No duplicate directed pair paths were detected.")
        lines.append("- Score CSV path/label/split rows align with the selected pair CSVs.")
    else:
        max_rows = 25
        lines.extend(["| source | row | pair set | kind | details |", "| --- | ---: | --- | --- | --- |"])
        for _, row in failures.head(max_rows).iterrows():
            lines.append(
                f"| `{Path(str(row['source_file'])).name}` | {row['row_index']} | "
                f"{row['pair_set']} | {row['kind']} | {row['details']} |"
            )
        if len(failures) > max_rows:
            lines.append(f"| ... | ... | ... | ... | {len(failures) - max_rows} more failures in CSV |")
    return "\n".join(lines) + "\n"


def run_validation(benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR, outdir: str | Path = DEFAULT_OUTDIR) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    output = parse_file_uri(outdir)
    checked_files: list[Path] = []
    issues: list[dict[str, Any]] = []

    selected_by_set: dict[str, pd.DataFrame | None] = {}
    for pair_set in PAIR_SETS:
        selected_path = benchmark / "selected_pairs" / f"{pair_set}.csv"
        checked_files.append(selected_path)
        selected = _read_csv(selected_path)
        selected_by_set[pair_set] = selected
        if selected is None:
            _issue(
                issues,
                source_file=selected_path,
                row_index=None,
                pair_set=pair_set,
                kind="file_exists",
                status="fail",
                details="selected pair CSV missing",
            )
        else:
            _validate_pair_rows(selected_path, pair_set, selected, issues)

    for method in METHODS:
        for pair_set in PAIR_SETS:
            score_path = benchmark / f"scores_{method}_{pair_set}.csv"
            selected_path = benchmark / "selected_pairs" / f"{pair_set}.csv"
            checked_files.append(score_path)
            _validate_alignment(
                selected_path=selected_path,
                selected=selected_by_set[pair_set],
                score_path=score_path,
                scores=_read_csv(score_path),
                pair_set=pair_set,
                rows=issues,
            )

    if not issues:
        for pair_set in PAIR_SETS:
            _issue(
                issues,
                source_file=benchmark,
                row_index=None,
                pair_set=pair_set,
                kind="all_checks",
                status="ok",
                details="no failures",
            )

    report = pd.DataFrame(issues)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "label_path_validation.csv"
    md_path = output / "label_path_validation.md"
    report.to_csv(csv_path, index=False)
    md_path.write_text(_render_markdown(benchmark_dir=benchmark, report=report, checked_files=checked_files), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate professor selected-pair labels against NIST SD300B path names.")
    parser.add_argument("--benchmark_dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_validation(args.benchmark_dir, args.outdir)
    print("Wrote label/path validation:")
    for path in paths.values():
        print(f"  {path}")
    report = pd.read_csv(paths["csv"])
    failures = int((report["status"] == "fail").sum()) if "status" in report.columns else 0
    print(f"Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
