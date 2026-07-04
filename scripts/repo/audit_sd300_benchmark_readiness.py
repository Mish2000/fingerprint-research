from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# These audits only need pandas' CSV/DataFrame basics. Avoid importing optional
# acceleration modules that can be stale in local NumPy 2 environments.
for _optional_pandas_module in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_optional_pandas_module, None)

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("nist_sd300b", "nist_sd300c")
EXPECTED = {
    "plain_rows": 8788,
    "roll_rows": 8871,
    "positive_pairs": 8779,
    "negative_pairs": 26337,
    "splits": {
        "train": {"pair_count": 28052, "positive_count": 7013, "negative_count": 21039},
        "val": {"pair_count": 3508, "positive_count": 877, "negative_count": 2631},
        "test": {"pair_count": 3556, "positive_count": 889, "negative_count": 2667},
    },
}
OLD_COUNT_PATTERNS = ("7029", "21087", "2812", "2844", "703", "2109", "711", "2133")
OLD_COUNT_TOKEN_RE = re.compile(
    r"(?<![0-9.])(?:" + "|".join(re.escape(value) for value in OLD_COUNT_PATTERNS) + r")(?![0-9.])"
)
CODE_PATH_PREFIXES = ("pipelines/", "scripts/", "src/", "tests/")


@dataclass
class Audit:
    root: Path
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")


def _run_git(root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _pairs_path(base: Path, split: str) -> Path:
    for candidate in (base / f"pairs_{split}.csv", base / "pairs" / f"pairs_{split}.csv"):
        if candidate.exists():
            return candidate
    return base / f"pairs_{split}.csv"


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    return {
        "pair_count": int(len(df)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }


def _frgp_values(df: pd.DataFrame) -> set[int]:
    return set(pd.to_numeric(df["frgp"], errors="coerce").dropna().astype(int).tolist())


def _check_manifest(audit: Audit, dataset: str, base: Path) -> pd.DataFrame | None:
    manifest_path = base / "manifest.csv"
    if not manifest_path.exists():
        audit.fail(f"{dataset}: missing manifest.csv at {manifest_path}")
        return None
    manifest = pd.read_csv(manifest_path)
    required = {"capture", "frgp", "path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        audit.fail(f"{dataset}: manifest.csv missing required readiness columns: {missing}")
        return None
    capture = manifest["capture"].astype(str).str.lower()
    plain_rows = int((capture == "plain").sum())
    roll_rows = int((capture == "roll").sum())
    if plain_rows != EXPECTED["plain_rows"] or roll_rows != EXPECTED["roll_rows"]:
        audit.fail(
            f"{dataset}: manifest plain/roll counts mismatch: "
            f"plain={plain_rows}, roll={roll_rows}, expected={EXPECTED['plain_rows']}/{EXPECTED['roll_rows']}"
        )
    else:
        audit.ok(f"{dataset}: manifest plain={plain_rows} roll={roll_rows}")
    coverage = sorted(_frgp_values(manifest))
    if coverage != list(range(1, 11)):
        audit.fail(f"{dataset}: manifest frgp coverage is {coverage}, expected 1..10")
    else:
        audit.ok(f"{dataset}: manifest anatomical frgp coverage 1..10")
    raw_available = "raw_frgp" in manifest.columns
    if not raw_available:
        audit.warn(f"{dataset}: raw_frgp column is not available in manifest.csv")
    return manifest


def _check_pair_file(audit: Audit, dataset: str, split: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        audit.fail(f"{dataset}/{split}: missing pair CSV at {path}")
        return None
    pairs = pd.read_csv(path)
    missing = sorted({"pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"} - set(pairs.columns))
    if missing:
        audit.fail(f"{dataset}/{split}: pair CSV missing columns: {missing}")
        return None
    counts = _label_counts(pairs)
    expected = EXPECTED["splits"][split]
    if counts != expected:
        audit.fail(f"{dataset}/{split}: counts mismatch expected={expected} actual={counts}")
    else:
        audit.ok(f"{dataset}/{split}: counts valid {counts}")
    frgps = _frgp_values(pairs)
    invalid = sorted(value for value in frgps if value < 1 or value > 10)
    if invalid:
        audit.fail(f"{dataset}/{split}: non-anatomical frgp values in pairs: {invalid}")
    refs = "\n".join(pairs["path_a"].astype(str).tolist() + pairs["path_b"].astype(str).tolist())
    if "plain_13" in refs or "plain_14" in refs:
        audit.fail(f"{dataset}/{split}: pair refs contain plain_13/plain_14")
    return pairs


def _check_total_pair_files(audit: Audit, dataset: str, base: Path) -> None:
    pos_path = base / "pairs_pos.csv"
    neg_path = base / "pairs_neg.csv"
    if not pos_path.exists() or not neg_path.exists():
        audit.fail(f"{dataset}: missing pairs_pos.csv or pairs_neg.csv")
        return
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)
    if len(pos) != EXPECTED["positive_pairs"] or len(neg) != EXPECTED["negative_pairs"]:
        audit.fail(
            f"{dataset}: positive/negative total mismatch: "
            f"pos={len(pos)}, neg={len(neg)}, expected={EXPECTED['positive_pairs']}/{EXPECTED['negative_pairs']}"
        )
    else:
        audit.ok(f"{dataset}: total positive/negative pair counts valid")
    for label, frame in (("positive", pos), ("negative", neg)):
        frgps = _frgp_values(frame)
        missing_thumbs = sorted({1, 6} - frgps)
        if missing_thumbs:
            audit.fail(f"{dataset}: {label} pairs missing anatomical thumb frgp values {missing_thumbs}")


def _tracked_files(root: Path) -> list[str]:
    return _run_git(root, ["ls-files"])


def _grep_tracked(root: Path, patterns: Iterable[str]) -> list[str]:
    expression = "|".join(patterns)
    return _run_git(root, ["grep", "-n", "-I", "-E", expression, "--", "."])


def _check_artifact_selected_pairs(audit: Audit, tracked: list[str]) -> None:
    stale = [
        path
        for path in tracked
        if path.startswith("artifacts/reports/")
        and "/selected_pairs/" in path.replace("\\", "/")
        and path.lower().endswith(".csv")
    ]
    if stale:
        audit.warn(f"tracked artifacts/reports selected_pairs CSVs are stale documentation output only: {len(stale)} file(s)")
    else:
        audit.ok("no tracked artifacts/reports selected_pairs CSVs found")


def _check_runner_sources(audit: Audit, root: Path) -> None:
    risky = []
    for line in _grep_tracked(
        root,
        [
            "plain_roll_final_baselines_v1/selected_pairs",
            "DEFAULT_SELECTED_PAIRS_DIR.*artifacts",
            "artifacts.*/selected_pairs",
        ],
    ):
        path = line.split(":", 1)[0].replace("\\", "/")
        if path == "scripts/repo/audit_sd300_benchmark_readiness.py":
            continue
        if not path.startswith(("pipelines/", "scripts/")):
            continue
        if "Refusing to use artifacts/reports" in line:
            continue
        risky.append(line)
    if risky:
        audit.fail("runners still reference artifacts/reports/**/selected_pairs as source input:\n  " + "\n  ".join(risky[:20]))
    else:
        audit.ok("runners do not use artifacts/reports/**/selected_pairs as source input")


def _check_old_counts(audit: Audit, root: Path) -> None:
    hits = []
    for line in _grep_tracked(root, OLD_COUNT_PATTERNS):
        path = line.split(":", 1)[0].replace("\\", "/")
        if path == "scripts/repo/audit_sd300_benchmark_readiness.py":
            continue
        if path.startswith(CODE_PATH_PREFIXES):
            content = line.split(":", 2)[-1]
            if OLD_COUNT_TOKEN_RE.search(content):
                hits.append(line)
    if hits:
        audit.fail("tracked code/test files still contain legacy SD300 counts:\n  " + "\n  ".join(hits[:30]))
    else:
        audit.ok("no legacy SD300 counts found in tracked code/test files")


def audit_repository(root: Path) -> int:
    audit = Audit(root=root)
    print(f"Repository root: {root}")
    for dataset in DATASETS:
        base = root / "data" / "manifests" / dataset
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset}")
        _check_manifest(audit, dataset, base)
        split_frames = []
        for split in ("train", "val", "test"):
            frame = _check_pair_file(audit, dataset, split, _pairs_path(base, split))
            if frame is not None:
                split_frames.append(frame)
        _check_total_pair_files(audit, dataset, base)
        if split_frames:
            all_pairs = pd.concat(split_frames, ignore_index=True, sort=False)
            coverage = sorted(_frgp_values(all_pairs))
            if coverage != list(range(1, 11)):
                audit.fail(f"{dataset}: pair frgp coverage is {coverage}, expected 1..10")
            else:
                audit.ok(f"{dataset}: pair frgp coverage 1..10")

    print("\n" + "=" * 80)
    tracked = _tracked_files(root)
    _check_artifact_selected_pairs(audit, tracked)
    _check_runner_sources(audit, root)
    _check_old_counts(audit, root)

    if audit.failures:
        print(f"\nAUDIT FAILED: {len(audit.failures)} failure(s), {len(audit.warnings)} warning(s)")
        return 1
    print(f"\nAUDIT PASSED: {len(audit.warnings)} warning(s)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SD300 benchmark/fusion readiness without modifying files.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return audit_repository(Path(args.root).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
