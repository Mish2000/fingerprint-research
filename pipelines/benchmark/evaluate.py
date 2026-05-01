from __future__ import annotations
import os
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# matplotlib for ROC plot
import matplotlib.pyplot as plt
import numpy as np
# sklearn is already used elsewhere in the repo (eval_classic.py)
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore

from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
)

METHOD_SEMANTICS_EPOCHS = {
    "harris": "harris_runtime_aligned_v1",
    "sift": "sift_runtime_aligned_v1",
}


# -----------------------------
# Helpers: path handling
# -----------------------------
def parse_file_uri(s: str) -> Path:
    # Accept: "file:/C:/path/..." or "C:/path/..." or relative path
    if s.startswith("file:"):
        s = s[len("file:"):]
        # "file:/C:/..." -> "/C:/..." on some inputs; strip leading slash if present
        if len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]
    return Path(s).expanduser().resolve()


def to_file_uri(p: Path) -> str:
    # Windows-friendly file URI style used in the project: file:/C:/...
    p = p.resolve()
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "file:/" + s
    return "file:" + s


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_dataset_dir(root: Path, dataset: str, input_dir: Optional[Path] = None) -> Path:
    """
    Supports both:
      - data/processed/<dataset>
      - data/manifests/<dataset>

    Returns the first directory that actually contains manifest.csv.
    """
    candidates: List[Path] = []

    if input_dir is not None:
        candidates.append(input_dir)

        try:
            parent_name = input_dir.parent.name.lower()
            if parent_name == "processed":
                candidates.append(input_dir.parent.parent / "manifests" / input_dir.name)
            elif parent_name == "manifests":
                candidates.append(input_dir.parent.parent / "processed" / input_dir.name)
        except Exception:
            pass

    candidates.append(root / "data" / "processed" / dataset)
    candidates.append(root / "data" / "manifests" / dataset)

    uniq_candidates: List[Path] = []
    seen: set[str] = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq_candidates.append(c)

    for c in uniq_candidates:
        if (c / "manifest.csv").exists():
            return c

    checked = [str(c) for c in uniq_candidates]
    raise FileNotFoundError(
        "Could not locate dataset directory containing manifest.csv. "
        f"Checked: {checked}"
    )


def resolve_pairs_path(data_dir: Path, split: str) -> Path:
    """
    Prefer flat canonical pairs_<split>.csv, but also support the nested compatibility copy at data_dir/pairs/pairs_<split>.csv.
    """
    candidates = [
        data_dir / f"pairs_{split}.csv",
        data_dir / "pairs" / f"pairs_{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # return preferred default path for clearer downstream messages
    return candidates[0]


# -----------------------------
# Metrics
# -----------------------------
def compute_auc_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if y_true.size == 0 or scores.size == 0:
        return float("nan"), float("nan")

    valid = np.isfinite(scores)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    scores = scores[valid]

    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")

    try:
        auc = float(roc_auc_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return float("nan"), float("nan")

    fnr = 1.0 - tpr
    delta = np.abs(fpr - fnr)
    if delta.size == 0 or np.isnan(delta).all():
        return auc, float("nan")

    i = int(np.nanargmin(delta))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return auc, eer


def tar_at_far(y_true: np.ndarray, scores: np.ndarray, far: float) -> float:
    """
    TAR@FAR: TAR = TPR at the largest threshold whose FPR <= far.
    If no such point exists (extreme far), fall back to nearest point.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if y_true.size == 0 or scores.size == 0:
        return float("nan")

    valid = np.isfinite(scores)
    if not np.any(valid):
        return float("nan")

    y_true = y_true[valid]
    scores = scores[valid]

    if np.unique(y_true).size < 2:
        return float("nan")

    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return float("nan")

    finite = np.isfinite(fpr) & np.isfinite(tpr)
    if not np.any(finite):
        return float("nan")

    fpr = fpr[finite]
    tpr = tpr[finite]

    idx = np.where(fpr <= far)[0]
    if len(idx) == 0:
        j = int(np.argmin(np.abs(fpr - far)))
        return float(tpr[j])
    return float(tpr[int(idx[-1])])


def save_roc_png(y_true: np.ndarray, scores: np.ndarray, out_png: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    ensure_parent(out_png)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.grid(True, which="both")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------------
# Standardized result row
# -----------------------------
@dataclass
class EvalRow:
    timestamp_utc: str
    method: str
    split: str
    n_pairs: int
    auc: float
    eer: float
    tar_at_far_1e_2: float
    tar_at_far_1e_3: float
    avg_ms_pair_reported: Optional[float]  # from method meta if available
    avg_ms_pair_wall: float  # wall clock / N (includes overhead)
    scores_csv: str
    meta_json: Optional[str]
    config_json: str


SUMMARY_HEADER = [
    "timestamp_utc",
    "method",
    "split",
    "n_pairs",
    "auc",
    "eer",
    "tar_at_far_1e_2",
    "tar_at_far_1e_3",
    "avg_ms_pair_reported",
    "avg_ms_pair_wall",
    "scores_csv",
    "meta_json",
    "config_json",
]


def utc_now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_scores(scores_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd  # local import to keep startup light
    df = pd.read_csv(scores_csv)
    if "label" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{scores_csv} must contain canonical score columns: label, score. Found: {list(df.columns)}")
    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    return y, s

def resolve_fusion_source_scores_path(source_dir: Path, dataset: str, method: str, split: str) -> Path:
    candidates = [
        source_dir / f"scores_{method}_{split}.csv",
        source_dir / f"scores_{dataset}_{method}_{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Fusion source scores not found for method={method!r}, split={split!r}. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def load_canonical_score_csv(path: Path, method_name: str):
    import pandas as pd
    df = pd.read_csv(path)
    required = {"label", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["label", "score"]].copy()
    out = out.rename(columns={"score": method_name})
    out["label"] = out["label"].astype(int)
    return out


def load_fusion_inputs(source_dir: Path, dataset: str, split: str, methods: List[str]):
    merged = None
    resolved_paths: Dict[str, str] = {}
    for method in methods:
        path = resolve_fusion_source_scores_path(source_dir, dataset, method, split)
        resolved_paths[method] = str(path)
        df = load_canonical_score_csv(path, method)
        if merged is None:
            merged = df
        else:
            if len(merged) != len(df):
                raise ValueError(f"Fusion row-count mismatch for {method} on split={split}")
            if not np.array_equal(merged["label"].values, df["label"].values):
                raise ValueError(f"Fusion label-order mismatch for {method} on split={split}")
            merged[method] = df[method].values
    assert merged is not None
    return merged, resolved_paths


def minmax_fit(series: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        x_max = x_min + 1e-12
    return x_min, x_max


def minmax_apply(series: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = (x - x_min) / (x_max - x_min)
    return np.clip(x, 0.0, 1.0)


def normalize_fusion_from_fit(fit_df, target_df, methods: List[str]):
    fit_norm: Dict[str, np.ndarray] = {}
    target_norm: Dict[str, np.ndarray] = {}
    for method in methods:
        x_min, x_max = minmax_fit(fit_df[method].values)
        fit_norm[method] = minmax_apply(fit_df[method].values, x_min, x_max)
        target_norm[method] = minmax_apply(target_df[method].values, x_min, x_max)
    return fit_norm, target_norm


def compute_weighted_fusion(norm_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return (
        weights["sift"] * norm_dict["sift"]
        + weights["dl_quick"] * norm_dict["dl_quick"]
        + weights["vit"] * norm_dict["vit"]
    )


def resolve_reference_summary_csv(source_dir: Path, dataset: str) -> Optional[Path]:
    candidates = [
        source_dir / "results_summary.csv",
        source_dir / f"results_summary_{dataset}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def estimate_fusion_avg_ms_pair(source_dir: Path, dataset: str, split: str, methods: List[str]) -> Optional[float]:
    summary_path = resolve_reference_summary_csv(source_dir, dataset)
    if summary_path is None:
        return None

    import pandas as pd
    df = pd.read_csv(summary_path)
    if "method" not in df.columns or "split" not in df.columns:
        return None

    sub = df[(df["split"] == split) & (df["method"].isin(methods))].copy()
    if sub.empty:
        return None

    if "avg_ms_pair_wall" in sub.columns:
        return float(pd.to_numeric(sub["avg_ms_pair_wall"], errors="coerce").fillna(0.0).sum())

    if "avg_ms_pair_reported" in sub.columns:
        return float(pd.to_numeric(sub["avg_ms_pair_reported"], errors="coerce").fillna(0.0).sum())

    return None


def maybe_load_meta(meta_path: Path) -> Dict[str, Any]:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def append_summary_row(summary_csv: Path, row: EvalRow) -> None:
    ensure_parent(summary_csv)
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(asdict(row))


# -----------------------------
# Runner: call existing scripts
# -----------------------------
def run_subprocess(cmd: List[str], *, cwd: Path) -> None:
    # Print the command for reproducibility
    print("\n[RUN]", " ".join(cmd))
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(cwd)
    subprocess.check_call(cmd, cwd=str(cwd), env=env)

def resolve_dedicated_ckpt_auto(root: Path) -> Optional[Path]:
    ckpt_dir = root / "artifacts" / "checkpoints"
    hits = list(ckpt_dir.rglob("patch_descriptor_ckpt.pth"))

    if not hits:
        hits = list(ckpt_dir.rglob("patch_descriptor/final/*.pth"))

    if not hits:
        return None

    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]

def main() -> None:
    ap = argparse.ArgumentParser(description="Unified evaluation orchestrator (Week 5).")
    ap.add_argument("--method", type=str, default="dl_quick",
                    choices=["classic_v2", "harris", "sift", "dl_quick", "dedicated", "vit", "fusion_balanced_v1"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only first N pairs (quick smoke tests).")
    ap.add_argument("--dataset", type=str, default="nist_sd300b",
                    help="Dataset name (e.g., nist_sd300b, nist_sd300c)")
    ap.add_argument("--data_dir", type=str, default="",
                    help="Optional dataset dir. Supports either data/processed/<dataset> or data/manifests/<dataset>.")

    ap.add_argument("--out_scores", type=str, default="", help="Optional explicit scores CSV output path.")
    ap.add_argument("--out_roc", type=str, default="", help="Optional explicit ROC PNG output path.")
    ap.add_argument("--out_run_meta", type=str, default="", help="Optional explicit run_meta JSON output path.")

    ap.add_argument("--summary_csv", type=str, default="",
                    help="Optional. If empty, defaults to artifacts/reports/benchmark/results_summary_<dataset>.csv")
    ap.add_argument("--script_classic", type=str, default="pipelines/benchmark/eval_classic.py")
    ap.add_argument("--script_dl", type=str, default="pipelines/benchmark/eval_quick.py")
    ap.add_argument("--script_dedicated", type=str, default="pipelines/benchmark/eval_dedicated.py")

    # Classic options
    ap.add_argument("--detector", type=str, default="gftt_orb", choices=["orb", "gftt_orb", "harris_orb", "sift"])
    ap.add_argument("--score_mode", type=str, default="inliers_over_k",
                    choices=["inliers_over_k", "inliers", "matches", "inliers_over_matches", "inliers_over_min_keypoints"])
    ap.add_argument("--nfeatures", type=int, default=1500)
    ap.add_argument("--long_edge", type=int, default=512)
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=4.0)

    # DL options
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50", "vit_base"])
    ap.add_argument("--no_mask", action="store_true", help="Disable mask pipeline for DL quick eval.")
    ap.add_argument("--emb_cache_dir", type=str, default="",
                    help="Persistent embedding cache dir (passed to eval_quick.py).")
    ap.add_argument("--cache_write", action="store_true",
                    help="Write missing embeddings into the disk cache.")
    ap.add_argument("--cache_strip_prefix", type=str, default="",
                    help="Optional prefix stripping for portable cache keys.")
    ap.add_argument("--ensure_pairs", action="store_true",
                    help="If pairs_<split>.csv are missing, regenerate them via pipelines/ingest/generate_pairs.py")

    # Dedicated options (Week 8)
    ap.add_argument("--dedicated_patch", type=int, default=48)
    ap.add_argument("--dedicated_max_kpts", type=int, default=800)
    ap.add_argument("--dedicated_mask_cov_thr", type=float, default=0.70)
    ap.add_argument("--dedicated_max_matches", type=int, default=200)
    ap.add_argument("--dedicated_ransac_thresh", type=float, default=4.0)
    ap.add_argument("--dedicated_ckpt", type=str, default="", help="Optional explicit descriptor ckpt path.")

    # Fusion options
    ap.add_argument("--fusion_source_dir", type=str, default="",
                    help="Directory containing source score CSVs for fusion.")
    ap.add_argument("--fusion_fit_split", type=str, default="val", choices=["train", "val"],
                    help="Split used to fit score normalization for fusion.")
    ap.add_argument("--fusion_sift_weight", type=float, default=0.91)
    ap.add_argument("--fusion_dl_weight", type=float, default=0.05)
    ap.add_argument("--fusion_vit_weight", type=float, default=0.04)

    args = ap.parse_args()

    root = Path.cwd().resolve()

    input_dir = parse_file_uri(args.data_dir) if args.data_dir else None
    data_dir = resolve_dataset_dir(root, args.dataset, input_dir)
    manifest_path = data_dir / "manifest.csv"
    pairs_path = resolve_pairs_path(data_dir, args.split)

    if args.ensure_pairs and not pairs_path.exists():
        gen_script = root / "pipelines" / "ingest" / "generate_pairs.py"
        cmd_gen = [sys.executable, str(gen_script), "--dataset", args.dataset, "--overwrite"]
        if args.data_dir:
            cmd_gen += ["--data_dir", str(data_dir)]
        run_subprocess(cmd_gen, cwd=root)
        pairs_path = resolve_pairs_path(data_dir, args.split)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    summary_csv = parse_file_uri(args.summary_csv) if args.summary_csv else (
        (root / "artifacts" / "reports" / "benchmark" / f"results_summary_{args.dataset}.csv").resolve()
    )

    out_dir = root / "artifacts" / "reports" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scores = parse_file_uri(args.out_scores) if args.out_scores else (
        out_dir / f"scores_{args.dataset}_{args.method}_{args.split}.csv"
    )
    out_roc = parse_file_uri(args.out_roc) if args.out_roc else (
        out_dir / f"roc_{args.dataset}_{args.method}_{args.split}.png"
    )
    out_run_meta = parse_file_uri(args.out_run_meta) if args.out_run_meta else (
        out_dir / f"run_{args.dataset}_{args.method}_{args.split}.meta.json"
    )

    ensure_parent(out_scores)
    ensure_parent(out_roc)
    ensure_parent(out_run_meta)

    # Build command
    t0 = time.time()

    resolved_detector = args.detector
    resolved_score_mode = args.score_mode
    resolved_ransac_thresh = args.ransac_thresh
    resolved_target_size = args.target_size
    resolved_method_semantics_epoch = None

    if args.method in ("classic_v2", "harris", "sift"):
        script = parse_file_uri(args.script_classic)
        if args.method == "classic_v2":
            resolved_detector = "gftt_orb"
            resolved_score_mode = "inliers_over_k"
            resolved_ransac_thresh = 4.0
        elif args.method == "harris":
            resolved_detector = "harris_orb"
            resolved_score_mode = "inliers_over_min_keypoints"
            resolved_ransac_thresh = 3.0
            resolved_target_size = 512
            resolved_method_semantics_epoch = METHOD_SEMANTICS_EPOCHS["harris"]
        elif args.method == "sift":
            resolved_detector = "sift"
            resolved_score_mode = "inliers_over_min_keypoints"
            resolved_ransac_thresh = 3.0
            resolved_target_size = 512
            resolved_method_semantics_epoch = METHOD_SEMANTICS_EPOCHS["sift"]
        cmd = [
            sys.executable, str(script),
            to_file_uri(out_scores),
            "--split", args.split,
            "--pairs", str(pairs_path),
            "--detector", resolved_detector,
            "--score_mode", resolved_score_mode,
            "--nfeatures", str(args.nfeatures),
            "--long_edge", str(args.long_edge),
            "--target_size", str(resolved_target_size),
            "--ratio", str(args.ratio),
            "--ransac_thresh", str(resolved_ransac_thresh),
            "--limit", str(args.limit),
        ]
        run_subprocess(cmd, cwd=root)
        meta_path = Path(str(out_scores) + ".meta.json")

    elif args.method in ["dl_quick", "vit"]:
        script = parse_file_uri(args.script_dl)

        # If method is vit, force the backbone to vit_base
        backbone_to_use = "vit_base" if args.method == "vit" else args.backbone

        cmd = [
            sys.executable, str(script),
            to_file_uri(out_scores),
            "--split", args.split,
            "--pairs", str(pairs_path),
            "--limit", str(args.limit),
            "--backbone", backbone_to_use,
            "--dataset", args.dataset,
        ]
        if args.emb_cache_dir:
            cmd += ["--emb_cache_dir", args.emb_cache_dir]
        if args.cache_write:
            cmd.append("--cache_write")
        if args.cache_strip_prefix:
            cmd += ["--cache_strip_prefix", args.cache_strip_prefix]
        if args.no_mask:
            cmd.append("--no_mask")
        run_subprocess(cmd, cwd=root)
        meta_path = out_scores.with_suffix(".meta.json")

    elif args.method == "dedicated":
        script = parse_file_uri(args.script_dedicated)

        cmd = [
            sys.executable, str(script),
            to_file_uri(out_scores),
            "--split", args.split,
            "--pairs", str(pairs_path),
            "--manifest", str(manifest_path),
            "--limit", str(args.limit),
            "--patch", str(args.dedicated_patch),
            "--max_kpts", str(args.dedicated_max_kpts),
            "--mask_cov_thr", str(args.dedicated_mask_cov_thr),
            "--max_matches", str(args.dedicated_max_matches),
            "--ransac_thresh", str(args.dedicated_ransac_thresh),
        ]

        resolved_dedicated_ckpt: Optional[Path] = None
        if args.dedicated_ckpt:
            resolved_dedicated_ckpt = parse_file_uri(args.dedicated_ckpt)
        else:
            resolved_dedicated_ckpt = resolve_dedicated_ckpt_auto(root)

        if resolved_dedicated_ckpt is None or not resolved_dedicated_ckpt.exists():
            raise FileNotFoundError(
                "Dedicated checkpoint not found. "
                "Pass --dedicated_ckpt explicitly or place the checkpoint under "
                "artifacts/checkpoints/**/patch_descriptor_ckpt.pth"
            )

        cmd += ["--ckpt", str(resolved_dedicated_ckpt)]
        run_subprocess(cmd, cwd=root)
        meta_path = out_scores.with_suffix(".meta.json")
    elif args.method == "fusion_balanced_v1":
        import pandas as pd

        source_methods = ["sift", "dl_quick", "vit"]
        weights = {
            "sift": float(args.fusion_sift_weight),
            "dl_quick": float(args.fusion_dl_weight),
            "vit": float(args.fusion_vit_weight),
        }

        if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
            raise ValueError(f"Fusion weights must sum to 1.0 exactly; got {weights}")

        source_dir = parse_file_uri(args.fusion_source_dir) if args.fusion_source_dir else out_scores.parent
        fit_split = args.fusion_fit_split

        fit_df, fit_paths = load_fusion_inputs(source_dir, args.dataset, fit_split, source_methods)
        target_df, target_paths = load_fusion_inputs(source_dir, args.dataset, args.split, source_methods)

        _, target_norm = normalize_fusion_from_fit(fit_df, target_df, source_methods)
        fused_scores = compute_weighted_fusion(target_norm, weights)

        pd.DataFrame({
            "label": target_df["label"].astype(int).values,
            "score": fused_scores,
        }).to_csv(out_scores, index=False)

        fusion_avg_ms_pair = estimate_fusion_avg_ms_pair(source_dir, args.dataset, args.split, source_methods)

        meta_path = out_scores.with_suffix(".meta.json")
        fusion_meta = {
            "schema_version": "v1_fusion_score_meta",
            "method": args.method,
            "split": args.split,
            "fit_split": fit_split,
            "weights": weights,
            "normalization": "minmax_fit_on_fit_split_only",
            "source_dir": str(source_dir),
            "source_methods": source_methods,
            "source_scores_fit": fit_paths,
            "source_scores_target": target_paths,
            "avg_ms_pair": fusion_avg_ms_pair,
        }
        meta_path.write_text(json.dumps(fusion_meta, indent=2, ensure_ascii=False), encoding="utf-8")


    t1 = time.time()

    # Read scores + compute unified metrics
    y, s = read_scores(out_scores)
    n = int(len(y))
    wall_ms_pair = float((t1 - t0) * 1000.0 / max(n, 1))

    auc, eer = compute_auc_eer(y, s)
    tar_1e2 = tar_at_far(y, s, 1e-2)
    tar_1e3 = tar_at_far(y, s, 1e-3)

    # Load reported latency from meta if exists
    reported_avg = None
    meta_json_path_str = None
    meta_obj = maybe_load_meta(meta_path)
    if meta_obj:
        meta_json_path_str = str(meta_path)
        if "avg_ms_pair" in meta_obj:
            try:
                reported_avg = float(meta_obj["avg_ms_pair"])
            except Exception:
                reported_avg = None

    # Save ROC png
    save_roc_png(y, s, out_roc, title=f"{args.method} | split={args.split} | AUC={auc:.4f} | EER~{eer:.4f}")

    # Create and append standardized row
    classic_config = {
        "script": str(args.script_classic),
        "detector": resolved_detector,
        "score_mode": resolved_score_mode,
        "nfeatures": args.nfeatures,
        "ratio": args.ratio,
        "ransac_thresh": resolved_ransac_thresh,
    }
    if args.method == "classic_v2":
        classic_config.update(
            {
                "long_edge": args.long_edge,
                "preprocess": "benchmark_long_edge_clahe_texture_roi",
                "mask_mode": "benchmark_texture_roi",
                "geometry_model": "affine_partial_2d",
                "normalization": "configured_nfeatures",
            }
        )
    elif args.method in {"harris", "sift"}:
        classic_config.update(
            {
                "target_size": resolved_target_size,
                "preprocess": "runtime_square_512_clahe_blur",
                "mask_mode": "none",
                "geometry_model": "homography",
                "normalization": "min_detected_keypoints",
            }
        )
        if resolved_method_semantics_epoch is not None:
            classic_config["method_semantics_epoch"] = resolved_method_semantics_epoch

    config = {
        "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "method": args.method,
        "split": args.split,
        "limit": args.limit,
        "dataset": args.dataset,
        "resolved_data_dir": str(data_dir),
        "manifest_path": str(manifest_path),
        "pairs_path": str(pairs_path),
        "method_semantics_epoch": resolved_method_semantics_epoch,
        "classic": classic_config,
        "dl": {
            "script": str(args.script_dl),
            "backbone": args.backbone,
            "no_mask": bool(args.no_mask),
        },
        "dedicated": {
            "script": str(args.script_dedicated),
            "patch": args.dedicated_patch,
            "max_kpts": args.dedicated_max_kpts,
            "mask_cov_thr": args.dedicated_mask_cov_thr,
            "max_matches": args.dedicated_max_matches,
            "ransac_thresh": args.dedicated_ransac_thresh,
            "ckpt": args.dedicated_ckpt if args.dedicated_ckpt else "auto",
        },
        "fusion": {
            "source_dir": args.fusion_source_dir if args.fusion_source_dir else str(out_scores.parent),
            "fit_split": args.fusion_fit_split,
            "weights": {
                "sift": args.fusion_sift_weight,
                "dl_quick": args.fusion_dl_weight,
                "vit": args.fusion_vit_weight,
            },
            "source_methods": ["sift", "dl_quick", "vit"],
        },
    }

    row = EvalRow(
        timestamp_utc=utc_now_iso(),
        method=args.method,
        split=args.split,
        n_pairs=n,
        auc=float(auc),
        eer=float(eer),
        tar_at_far_1e_2=float(tar_1e2),
        tar_at_far_1e_3=float(tar_1e3),
        avg_ms_pair_reported=reported_avg,
        avg_ms_pair_wall=float(wall_ms_pair),
        scores_csv=str(out_scores),
        meta_json=meta_json_path_str,
        config_json=json.dumps(config, ensure_ascii=False),
    )
    append_summary_row(summary_csv, row)

    # Save run meta (full, readable)
    run_meta = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": asdict(row),
        "scores_csv": str(out_scores),
        "roc_png": str(out_roc),
        "summary_csv": str(summary_csv),
        "method_meta_json": meta_json_path_str,
        "resolved_data_dir": str(data_dir),
        "manifest_path": str(manifest_path),
        "pairs_path": str(pairs_path),
        "config": config,
    }
    out_run_meta.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[OK] Unified metrics:")
    print(f"  N={n} | AUC={auc:.4f} | EER~{eer:.4f} | TAR@1e-2={tar_1e2:.4f} | TAR@1e-3={tar_1e3:.4f}")
    print("  DataDir :", data_dir)
    print("  Manifest:", manifest_path)
    print("  Pairs   :", pairs_path)
    print("  Scores  :", out_scores)
    print("  ROC     :", out_roc)
    print("  Summary :", summary_csv)
    print("  RunMeta :", out_run_meta)


if __name__ == "__main__":
    main()
