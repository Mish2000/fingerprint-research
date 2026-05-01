from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from importlib.metadata import version as pkg_version  # type: ignore
except Exception:
    pkg_version = None  # type: ignore


FUSION_METHOD = "fusion_balanced_v1"
FUSION_SOURCE_METHODS = ["sift", "dl_quick", "vit"]


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
EVALUATE_PY = ROOT / "pipelines" / "benchmark" / "evaluate.py"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_log_line(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def resolve_path(raw: str) -> Path:
    if raw.startswith("file:"):
        raw = raw[len("file:"):]
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_cmd_stream(cmd: List[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(ROOT)
    env.setdefault("PYTHONHASHSEED", "0")

    with subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            write_log_line(log_path, line.rstrip("\n"))
        return proc.wait()


@dataclass
class GitInfo:
    commit: Optional[str]
    is_dirty: Optional[bool]
    branch: Optional[str]
    error: Optional[str]


def get_git_info(repo_root: Path) -> GitInfo:
    try:
        rc, out, err = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if rc != 0:
            return GitInfo(None, None, None, f"git rev-parse failed: {err or out}")

        commit = out

        rc, out, err = run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
        if rc != 0:
            return GitInfo(commit, None, None, f"git status failed: {err or out}")
        is_dirty = len(out.strip()) > 0

        rc, out, err = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        if rc != 0:
            return GitInfo(commit, is_dirty, None, f"git branch failed: {err or out}")

        return GitInfo(commit, is_dirty, out.strip(), None)
    except FileNotFoundError:
        return GitInfo(None, None, None, "git not found on PATH")
    except Exception as exc:
        return GitInfo(None, None, None, f"git info error: {exc}")


def safe_pkg_version(name: str) -> Optional[str]:
    if pkg_version is None:
        return None
    try:
        return pkg_version(name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_dir(dataset: str, user_data_dir: str = "") -> Path:
    candidates: List[Path] = []

    if user_data_dir:
        path = resolve_path(user_data_dir)
        candidates.append(path)

        try:
            parent_name = path.parent.name.lower()
            if parent_name == "processed":
                candidates.append(path.parent.parent / "manifests" / path.name)
            elif parent_name == "manifests":
                candidates.append(path.parent.parent / "processed" / path.name)
        except Exception:
            pass

    candidates.append(ROOT / "data" / "processed" / dataset)
    candidates.append(ROOT / "data" / "manifests" / dataset)

    unique_candidates: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        raw = str(candidate)
        if raw not in seen:
            seen.add(raw)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if (candidate / "manifest.csv").exists():
            return candidate

    checked = [str(candidate) for candidate in unique_candidates]
    raise FileNotFoundError(
        "Could not locate dataset directory containing manifest.csv. "
        f"Checked: {checked}"
    )


def required_files_for_data_dir(data_dir: Path) -> List[Path]:
    return [
        data_dir / "split.json",
        data_dir / "stats.json",
    ]


def optional_files_for_data_dir(data_dir: Path) -> List[Path]:
    return [
        data_dir / "pairs_split_build.meta.json",
    ]


def resolve_dedicated_ckpt(user_value: str) -> str:
    if user_value and user_value.lower() != "auto":
        return str(resolve_path(user_value))

    ckpt_dir = ROOT / "artifacts" / "checkpoints"
    hits = list(ckpt_dir.rglob("patch_descriptor_ckpt.pth"))

    if not hits:
        hits = list(ckpt_dir.rglob("patch_descriptor/final/*.pth"))

    if not hits:
        raise FileNotFoundError(
            "Could not auto-resolve dedicated checkpoint. "
            "Expected something like artifacts/checkpoints/**/patch_descriptor_ckpt.pth"
        )

    hits.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(hits[0])


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def validate_fusion_request(methods: list[str], splits: list[str], fusion_fit_split: str) -> None:
    if FUSION_METHOD not in methods:
        return

    missing_sources = [method for method in FUSION_SOURCE_METHODS if method not in methods]
    if missing_sources:
        raise ValueError(
            f"{FUSION_METHOD} requires source methods in the same batch request. "
            f"Missing: {missing_sources}"
        )

    if fusion_fit_split not in splits:
        raise ValueError(
            f"{FUSION_METHOD} requires fusion_fit_split={fusion_fit_split!r} "
            f"to be present in requested splits {splits}"
        )

    fusion_index = methods.index(FUSION_METHOD)
    for source_method in FUSION_SOURCE_METHODS:
        if methods.index(source_method) > fusion_index:
            raise ValueError(
                f"{FUSION_METHOD} must appear after its source methods. "
                f"Found method order: {methods}"
            )


def build_eval_cmd(
    *,
    outdir: Path,
    dataset: str,
    data_dir: Path,
    method: str,
    split: str,
    limit: int,
    ensure_pairs: bool,
    dedicated_ckpt: str,
    emb_cache_dir: str = "",
    cache_write: bool = False,
    cache_strip_prefix: str = "",
    fusion_fit_split: str = "val",
    fusion_sift_weight: float = 0.91,
    fusion_dl_weight: float = 0.05,
    fusion_vit_weight: float = 0.04,
) -> List[str]:
    summary_csv = outdir / "results_summary.csv"
    out_scores = outdir / f"scores_{method}_{split}.csv"
    out_roc = outdir / f"roc_{method}_{split}.png"
    out_run_meta = outdir / f"run_{method}_{split}.meta.json"

    cmd = [
        sys.executable,
        str(EVALUATE_PY),
        "--method", method,
        "--split", split,
        "--limit", str(limit),
        "--dataset", dataset,
        "--data_dir", str(data_dir),
        "--summary_csv", str(summary_csv),
        "--out_scores", str(out_scores),
        "--out_roc", str(out_roc),
        "--out_run_meta", str(out_run_meta),
    ]

    if ensure_pairs:
        cmd.append("--ensure_pairs")

    if method == "classic_v2":
        cmd += [
            "--detector", "gftt_orb",
            "--score_mode", "inliers_over_k",
            "--nfeatures", "1500",
            "--long_edge", "512",
            "--ratio", "0.75",
            "--ransac_thresh", "4.0",
        ]
    elif method == "harris":
        cmd += [
            "--detector", "harris_orb",
            "--score_mode", "inliers_over_min_keypoints",
            "--nfeatures", "1500",
            "--target_size", "512",
            "--ratio", "0.75",
            "--ransac_thresh", "3.0",
        ]
    elif method == "sift":
        cmd += [
            "--detector", "sift",
            "--score_mode", "inliers_over_min_keypoints",
            "--nfeatures", "1500",
            "--target_size", "512",
            "--ratio", "0.75",
            "--ransac_thresh", "3.0",
        ]
    elif method in {"dl_quick", "vit"}:
        backbone = "vit_base" if method == "vit" else "resnet50"
        # Canonical benchmark defaults now match the shipped runtime behavior:
        # masking stays enabled unless --no_mask is passed explicitly elsewhere.
        cmd += ["--backbone", backbone]

        if emb_cache_dir:
            cmd += ["--emb_cache_dir", emb_cache_dir]
        if cache_write:
            cmd.append("--cache_write")
        if cache_strip_prefix:
            cmd += ["--cache_strip_prefix", cache_strip_prefix]
    elif method == "dedicated":
        cmd += [
            "--dedicated_patch", "48",
            "--dedicated_max_kpts", "800",
            "--dedicated_mask_cov_thr", "0.7",
            "--dedicated_max_matches", "200",
            "--dedicated_ransac_thresh", "4.0",
            "--dedicated_ckpt", resolve_dedicated_ckpt(dedicated_ckpt),
        ]
    elif method == FUSION_METHOD:
        cmd += [
            "--fusion_source_dir", str(outdir),
            "--fusion_fit_split", fusion_fit_split,
            "--fusion_sift_weight", str(fusion_sift_weight),
            "--fusion_dl_weight", str(fusion_dl_weight),
            "--fusion_vit_weight", str(fusion_vit_weight),
        ]
    else:
        raise ValueError(f"Unsupported method: {method}")

    return cmd


def render_results_md(summary_csv: Path, summary_md: Path) -> None:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("results_summary.csv exists but is empty")

    split_order = {"val": 0, "test": 1, "train": 2}
    method_order = {
        "classic_v2": 0,
        "harris": 1,
        "sift": 2,
        "dl_quick": 3,
        "dedicated": 4,
        "vit": 5,
        FUSION_METHOD: 6,
    }

    rows.sort(
        key=lambda row: (
            split_order.get(row.get("split", ""), 99),
            method_order.get(row.get("method", ""), 99),
            row.get("method", ""),
        )
    )

    cols = [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
    ]
    existing_cols = [col for col in cols if col in rows[0]]

    lines = [
        "# Benchmark Results Summary",
        "",
        "| " + " | ".join(existing_cols) + " |",
        "| " + " | ".join(["---"] * len(existing_cols)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in existing_cols) + " |")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_payload(
    *,
    dataset: str,
    data_dir: Path,
    outdir: Path,
    methods: list[str],
    splits: list[str],
    limit: int,
    ensure_pairs: bool,
    emb_cache_dir: str,
    cache_write: bool,
    cache_strip_prefix: str,
    dedicated_ckpt: str,
    fusion_fit_split: str,
    fusion_sift_weight: float,
    fusion_dl_weight: float,
    fusion_vit_weight: float,
    input_hashes: Dict[str, str],
    mode: str,
    argv: Optional[list[str]] = None,
) -> Dict[str, object]:
    packages = {
        "numpy": safe_pkg_version("numpy"),
        "opencv-python": safe_pkg_version("opencv-python"),
        "torch": safe_pkg_version("torch"),
        "torchvision": safe_pkg_version("torchvision"),
        "pandas": safe_pkg_version("pandas"),
        "fastapi": safe_pkg_version("fastapi"),
        "uvicorn": safe_pkg_version("uvicorn"),
    }

    payload: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repo_root": str(ROOT),
        "outdir": str(outdir),
        "dataset": {
            "name": dataset,
            "resolved_data_dir": str(data_dir),
        },
        "methods": methods,
        "splits": splits,
        "limit": int(limit),
        "ensure_pairs": bool(ensure_pairs),
        "dedicated_ckpt": dedicated_ckpt,
        "emb_cache_dir": emb_cache_dir,
        "cache_write": bool(cache_write),
        "cache_strip_prefix": cache_strip_prefix,
        "fusion": {
            "fit_split": fusion_fit_split,
            "weights": {
                "sift": float(fusion_sift_weight),
                "dl_quick": float(fusion_dl_weight),
                "vit": float(fusion_vit_weight),
            },
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "git": asdict(get_git_info(ROOT)),
        "packages": packages,
        "input_hashes": input_hashes,
    }

    if argv is not None:
        payload["argv"] = argv

    # Compatibility aliases preserved during the runner migration so older
    # downstream readers can continue consuming the canonical manifest.
    payload["package_versions"] = dict(packages)
    payload["file_hashes_sha256"] = dict(input_hashes)

    return payload


def expected_output_paths(outdir: Path, methods: list[str], splits: list[str]) -> list[Path]:
    expected = [
        outdir / "results_summary.csv",
        outdir / "results_summary.md",
        outdir / "run_manifest.json",
        outdir / "run.log",
    ]
    for split in splits:
        for method in methods:
            expected.extend([
                outdir / f"scores_{method}_{split}.csv",
                outdir / f"roc_{method}_{split}.png",
                outdir / f"run_{method}_{split}.meta.json",
            ])
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical batch benchmark runner.")
    parser.add_argument("--dataset", type=str, default="nist_sd300b")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--outdir", type=str, default="artifacts/reports/benchmark/full_nist_sd300b")
    parser.add_argument("--methods", type=str, default="classic_v2,harris,sift,dl_quick,dedicated,vit")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ensure_pairs", action="store_true")
    parser.add_argument("--dedicated_ckpt", type=str, default="auto")
    parser.add_argument("--emb_cache_dir", type=str, default="artifacts/cache/embeddings")
    parser.add_argument("--cache_write", action="store_true")
    parser.add_argument("--cache_strip_prefix", type=str, default="")
    parser.add_argument("--fusion_fit_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--fusion_sift_weight", type=float, default=0.91)
    parser.add_argument("--fusion_dl_weight", type=float, default=0.05)
    parser.add_argument("--fusion_vit_weight", type=float, default=0.04)
    return parser


def run_matrix(args: argparse.Namespace) -> int:
    outdir = resolve_path(args.outdir)
    ensure_dir(outdir)

    log_path = outdir / "run.log"
    if log_path.exists():
        log_path.unlink()

    def log(message: str) -> None:
        print(message)
        write_log_line(log_path, message)

    try:
        data_dir = resolve_dataset_dir(args.dataset, args.data_dir)
    except Exception as exc:
        log(f"ERROR: Could not resolve dataset dir: {exc}")
        return 2

    required_files = required_files_for_data_dir(data_dir)
    missing_required = [path for path in required_files if not path.exists()]
    if missing_required:
        log("ERROR: Missing required dataset metadata files:")
        for path in missing_required:
            log(f"  - {path}")
        return 2

    methods = parse_csv_list(args.methods)
    splits = parse_csv_list(args.splits)
    if not methods:
        log("ERROR: --methods resolved to an empty method list.")
        return 2
    if not splits:
        log("ERROR: --splits resolved to an empty split list.")
        return 2

    try:
        validate_fusion_request(methods, splits, args.fusion_fit_split)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 2

    optional_files = [path for path in optional_files_for_data_dir(data_dir) if path.exists()]
    input_hashes = {str(path): sha256_file(path) for path in required_files + optional_files}

    emb_cache_dir = str(resolve_path(args.emb_cache_dir)) if args.emb_cache_dir else ""
    dedicated_ckpt = (
        resolve_dedicated_ckpt(args.dedicated_ckpt)
        if "dedicated" in methods
        else args.dedicated_ckpt
    )

    manifest = build_manifest_payload(
        dataset=args.dataset,
        data_dir=data_dir,
        outdir=outdir,
        methods=methods,
        splits=splits,
        limit=args.limit,
        ensure_pairs=args.ensure_pairs,
        emb_cache_dir=emb_cache_dir,
        cache_write=args.cache_write,
        cache_strip_prefix=args.cache_strip_prefix,
        dedicated_ckpt=dedicated_ckpt,
        fusion_fit_split=args.fusion_fit_split,
        fusion_sift_weight=args.fusion_sift_weight,
        fusion_dl_weight=args.fusion_dl_weight,
        fusion_vit_weight=args.fusion_vit_weight,
        input_hashes=input_hashes,
        mode="batch",
        argv=list(sys.argv),
    )
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_csv = outdir / "results_summary.csv"
    summary_md = outdir / "results_summary.md"
    if summary_csv.exists():
        summary_csv.unlink()
    if summary_md.exists():
        summary_md.unlink()

    log("=== Canonical batch benchmark run ===")
    log(f"Repo root : {ROOT}")
    log(f"Dataset   : {args.dataset}")
    log(f"Data dir  : {data_dir}")
    log(f"Outdir    : {outdir}")
    log(f"Methods   : {methods}")
    log(f"Splits    : {splits}")
    if emb_cache_dir:
        log(f"Emb cache : {emb_cache_dir}")

    if not EVALUATE_PY.exists():
        log(f"ERROR: Missing evaluate.py -> {EVALUATE_PY}")
        return 3

    for split in splits:
        for method in methods:
            if method == "dedicated":
                log(f"[DEDICATED] using ckpt: {dedicated_ckpt}")

            cmd = build_eval_cmd(
                outdir=outdir,
                dataset=args.dataset,
                data_dir=data_dir,
                method=method,
                split=split,
                limit=args.limit,
                ensure_pairs=args.ensure_pairs,
                dedicated_ckpt=dedicated_ckpt,
                emb_cache_dir=emb_cache_dir,
                cache_write=args.cache_write,
                cache_strip_prefix=args.cache_strip_prefix,
                fusion_fit_split=args.fusion_fit_split,
                fusion_sift_weight=args.fusion_sift_weight,
                fusion_dl_weight=args.fusion_dl_weight,
                fusion_vit_weight=args.fusion_vit_weight,
            )
            log("[RUN] " + " ".join(cmd))
            rc = run_cmd_stream(cmd, cwd=ROOT, log_path=log_path)
            log(f"[DONE] method={method} split={split} exit_code={rc}")

            if rc != 0:
                log("ERROR: Stopped due to non-zero exit code.")
                return 4

    render_results_md(summary_csv, summary_md)
    log(f"OK: Wrote {summary_md}")

    missing_outputs = [path for path in expected_output_paths(outdir, methods, splits) if not path.exists()]
    if missing_outputs:
        log("ERROR: Missing expected outputs:")
        for path in missing_outputs:
            log(f"  - {path}")
        return 5

    log("Done.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
