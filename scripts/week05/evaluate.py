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

# matplotlib for ROC plot
import matplotlib.pyplot as plt
import numpy as np
# sklearn is already used elsewhere in the repo (week3_score_pairs_v2.py)
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore


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


# -----------------------------
# Metrics
# -----------------------------
def compute_auc_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return auc, eer


def tar_at_far(y_true: np.ndarray, scores: np.ndarray, far: float) -> float:
    """
    TAR@FAR: TAR = TPR at the largest threshold whose FPR <= far.
    If no such point exists (extreme far), fall back to nearest point.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(fpr <= far)[0]
    if len(idx) == 0:
        # pick closest FPR
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
        raise ValueError(f"{scores_csv} must contain columns: label, score. Found: {list(df.columns)}")
    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    return y, s


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified evaluation orchestrator (Week 5).")
    ap.add_argument("--method", type=str, default="dl_quick",
                    choices=["classic_v2", "dl_quick", "dedicated"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only first N pairs (quick smoke tests).")

    ap.add_argument("--summary_csv", type=str, default="reports/week05/results_summary.csv")
    ap.add_argument("--out_scores", type=str, default="", help="Optional explicit scores CSV output path.")
    ap.add_argument("--out_roc", type=str, default="", help="Optional explicit ROC PNG output path.")
    ap.add_argument("--out_run_meta", type=str, default="", help="Optional explicit run_meta JSON output path.")

    # Script paths (override if your repo paths differ)
    ap.add_argument("--script_classic", type=str, default="scripts/week03/week3_score_pairs_v2.py")
    ap.add_argument("--script_dl", type=str, default="scripts/week04/eval_quick.py")
    ap.add_argument("--script_dedicated", type=str, default="scripts/week08/eval_dedicated.py")

    # Classic options
    ap.add_argument("--detector", type=str, default="gftt_orb", choices=["orb", "gftt_orb"])
    ap.add_argument("--score_mode", type=str, default="inliers_over_k",
                    choices=["inliers_over_k", "inliers", "matches", "inliers_over_matches"])
    ap.add_argument("--nfeatures", type=int, default=1500)
    ap.add_argument("--long_edge", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=4.0)

    # DL options
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--no_mask", action="store_true", help="Disable mask pipeline for DL quick eval.")
    ap.add_argument("--ensure_pairs", action="store_true",
                    help="If pairs_<split>.csv are missing, regenerate them from pairs_pos/neg via week05/generate_pairs.py")

    # Dedicated options (Week 8)
    ap.add_argument("--dedicated_patch", type=int, default=48)
    ap.add_argument("--dedicated_max_kpts", type=int, default=800)
    ap.add_argument("--dedicated_mask_cov_thr", type=float, default=0.70)
    ap.add_argument("--dedicated_max_matches", type=int, default=200)
    ap.add_argument("--dedicated_ransac_thresh", type=float, default=4.0)
    ap.add_argument("--dedicated_ckpt", type=str, default="", help="Optional explicit descriptor ckpt path.")

    args = ap.parse_args()

    root = Path.cwd()

    data_dir = root / "data" / "processed" / "nist_sd300b"
    pairs_path = data_dir / f"pairs_{args.split}.csv"

    if args.ensure_pairs and not pairs_path.exists():
        gen_script = root / "scripts" / "week05" / "generate_pairs.py"
        cmd_gen = [sys.executable, str(gen_script)]
        run_subprocess(cmd_gen, cwd=root)

    summary_csv = parse_file_uri(args.summary_csv)

    # Default outputs
    out_dir = root / "reports" / "week05"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scores = parse_file_uri(args.out_scores) if args.out_scores else (
                out_dir / f"scores_{args.method}_{args.split}.csv")
    out_roc = parse_file_uri(args.out_roc) if args.out_roc else (out_dir / f"roc_{args.method}_{args.split}.png")
    out_run_meta = parse_file_uri(args.out_run_meta) if args.out_run_meta else (
                out_dir / f"run_{args.method}_{args.split}.meta.json")

    ensure_parent(out_scores)
    ensure_parent(out_roc)
    ensure_parent(out_run_meta)

    # Build command
    t0 = time.time()

    if args.method == "classic_v2":
        script = parse_file_uri(args.script_classic)
        cmd = [
            sys.executable, str(script),
            to_file_uri(out_scores),
            "--split", args.split,
            "--detector", args.detector,
            "--score_mode", args.score_mode,
            "--nfeatures", str(args.nfeatures),
            "--long_edge", str(args.long_edge),
            "--ratio", str(args.ratio),
            "--ransac_thresh", str(args.ransac_thresh),
            "--limit", str(args.limit),
        ]
        run_subprocess(cmd, cwd=root)
        meta_path = Path(str(out_scores) + ".meta.json")  # classic script might not create it; we handle missing
    elif args.method == "dl_quick":
        script = parse_file_uri(args.script_dl)
        cmd = [
            sys.executable, str(script),
            to_file_uri(out_scores),
            "--split", args.split,
            "--limit", str(args.limit),
            "--backbone", args.backbone,
        ]
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
            "--limit", str(args.limit),
            "--patch", str(args.dedicated_patch),
            "--max_kpts", str(args.dedicated_max_kpts),
            "--mask_cov_thr", str(args.dedicated_mask_cov_thr),
            "--max_matches", str(args.dedicated_max_matches),
            "--ransac_thresh", str(args.dedicated_ransac_thresh),
        ]

        # Prefer explicit ckpt if provided; otherwise pass a robust auto-detected default if it exists.
        if args.dedicated_ckpt:
            cmd += ["--ckpt", str(parse_file_uri(args.dedicated_ckpt))]
        else:
            ckpt_guess = root / "reports" / "week06+07" / "patch_descriptor" / "final" / "patch_descriptor_ckpt.pth"
            if ckpt_guess.exists():
                cmd += ["--ckpt", str(ckpt_guess)]

        run_subprocess(cmd, cwd=root)
        meta_path = out_scores.with_suffix(".meta.json")

    t1 = time.time()
    wall_ms_pair = None

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
    config = {
        "method": args.method,
        "split": args.split,
        "limit": args.limit,
        "classic": {
            "script": str(args.script_classic),
            "detector": args.detector,
            "score_mode": args.score_mode,
            "nfeatures": args.nfeatures,
            "long_edge": args.long_edge,
            "ratio": args.ratio,
            "ransac_thresh": args.ransac_thresh,
        },
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
        "row": asdict(row),
        "scores_csv": str(out_scores),
        "roc_png": str(out_roc),
        "summary_csv": str(summary_csv),
        "method_meta_json": meta_json_path_str,
    }
    out_run_meta.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[OK] Unified metrics:")
    print(f"  N={n} | AUC={auc:.4f} | EER~{eer:.4f} | TAR@1e-2={tar_1e2:.4f} | TAR@1e-3={tar_1e3:.4f}")
    print("  Scores :", out_scores)
    print("  ROC    :", out_roc)
    print("  Summary:", summary_csv)
    print("  RunMeta :", out_run_meta)


if __name__ == "__main__":
    main()
