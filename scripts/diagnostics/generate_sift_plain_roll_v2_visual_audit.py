from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.generate_plain_roll_visual_audit import (  # noqa: E402
    _fit_width,
    _label,
    _pad_to_width,
    _side_by_side,
)
from src.fpbench.matchers.matching_baseline import (  # noqa: E402
    ransac_inliers_for_model,
    score_sift_plain_roll_v2_counts,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, preprocess_image  # noqa: E402


DEFAULT_DIAG_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
DEFAULT_SCORE_DIR = DEFAULT_DIAG_DIR / "sift_plain_roll_v2_scores"
DEFAULT_OUTDIR = DEFAULT_DIAG_DIR / "sift_plain_roll_v2_visual_audit"
PAIR_SETS = ("positive_1000", "negative_1000")


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_gray(path_str: str) -> np.ndarray:
    path = parse_file_uri(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _load_scores(score_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pair_set in PAIR_SETS:
        path = score_dir / f"scores_sift_plain_roll_v2_{pair_set}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing score CSV: {path}")
        df = pd.read_csv(path)
        missing = {"label", "split", "path_a", "path_b", "score", "matches", "inliers"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["pair_set"] = pair_set
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["score"] = pd.to_numeric(combined["score"], errors="coerce").fillna(0.0)
    combined["matches"] = pd.to_numeric(combined["matches"], errors="coerce").fillna(0).astype(int)
    combined["inliers"] = pd.to_numeric(combined["inliers"], errors="coerce").fillna(0).astype(int)
    combined["inlier_ratio"] = combined["inliers"] / combined["matches"].clip(lower=1)
    return combined


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        actual_far = false_accepts / n_negative
        if actual_far <= float(target_far):
            return float(threshold), false_accepts, float(actual_far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _calibrate_threshold(scores: pd.DataFrame, target_far: float) -> tuple[float, int, float]:
    split = scores["split"].astype(str).str.strip().str.lower()
    labels = scores["label"].astype(int)
    negatives = scores.loc[(split == "val") & (labels == 0), "score"].to_numpy(dtype=float)
    return _threshold_for_far(negatives, float(target_far))


def _parse_filename_metadata(path_str: str) -> dict[str, str]:
    name = Path(str(path_str)).stem
    match = re.search(r"(?P<subject>\d+)_+(?P<capture>plain|roll).*?_(?P<frgp>\d+)$", name, flags=re.IGNORECASE)
    if not match:
        return {"subject": "", "capture": "", "frgp": ""}
    return {
        "subject": match.group("subject"),
        "capture": match.group("capture").lower(),
        "frgp": match.group("frgp"),
    }


def _case_slug(row: pd.Series, group_name: str, rank: int) -> str:
    payload = f"{row.get('path_a', '')}|{row.get('path_b', '')}|{group_name}|{rank}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{group_name}_{rank:02d}_{digest}"


def _sift_v2_views(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    target_size: int,
    nfeatures: int,
    blur_ksize: int,
    ratio: float,
    ransac_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    proc_a = preprocess_image(
        img_a,
        PreprocessConfig(target_size=int(target_size), blur_ksize=int(blur_ksize)),
    )
    proc_b = preprocess_image(
        img_b,
        PreprocessConfig(target_size=int(target_size), blur_ksize=int(blur_ksize)),
    )
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    kps_a, desc_a = sift.detectAndCompute(proc_a, None)
    kps_b, desc_b = sift.detectAndCompute(proc_b, None)
    kps_a = kps_a or []
    kps_b = kps_b or []
    key_a = cv2.drawKeypoints(proc_a, kps_a, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    key_b = cv2.drawKeypoints(proc_b, kps_b, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints = _side_by_side(
        key_a,
        key_b,
        f"SIFT v2 keypoints plain: {len(kps_a)}",
        f"SIFT v2 keypoints roll: {len(kps_b)}",
        height=320,
    )

    diagnostics: dict[str, Any] = {
        "k1_recomputed": len(kps_a),
        "k2_recomputed": len(kps_b),
        "matches_recomputed": 0,
        "inliers_recomputed": 0,
        "inlier_ratio_recomputed": 0.0,
        "score_recomputed": 0.0,
    }
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        blank = np.full((240, 900, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "No SIFT descriptors", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        return keypoints, _label(blank, "SIFT v2 Lowe-ratio matches"), _label(blank, "SIFT v2 affine_full_2d inliers"), diagnostics

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for item in knn:
        if len(item) == 2:
            first, second = item
            if first.distance < float(ratio) * second.distance:
                good.append(first)
    good_sorted = sorted(good, key=lambda m: float(m.distance))
    good_img = cv2.drawMatches(
        proc_a,
        kps_a,
        proc_b,
        kps_b,
        good_sorted[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    good_img = _label(_fit_width(good_img, 1500), f"SIFT v2 Lowe-ratio matches: {len(good)}")

    inliers, mask = ransac_inliers_for_model(
        kps_a,
        kps_b,
        good,
        ransac_model="affine_full_2d",
        ransac_thresh=float(ransac_thresh),
    )
    diagnostics.update(
        {
            "matches_recomputed": int(len(good)),
            "inliers_recomputed": int(inliers),
            "inlier_ratio_recomputed": float(inliers) / float(max(len(good), 1)),
            "score_recomputed": score_sift_plain_roll_v2_counts(matches=len(good), inliers=inliers),
        }
    )
    if mask is not None:
        inlier_matches = [m for m, keep in zip(good, mask.astype(bool).tolist()) if keep]
        inlier_matches = sorted(inlier_matches, key=lambda m: float(m.distance))[:80]
        raw = cv2.drawMatches(
            proc_a,
            kps_a,
            proc_b,
            kps_b,
            inlier_matches,
            None,
            matchColor=(0, 210, 0),
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        inlier_img = _label(_fit_width(raw, 1500), f"SIFT v2 affine_full_2d inliers: {int(inliers)}")
    else:
        blank = np.full((240, 900, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "Too few matches or no affine model", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        inlier_img = _label(blank, "SIFT v2 affine_full_2d inliers")
    return keypoints, good_img, inlier_img, diagnostics


def _make_case_sheet(row: pd.Series, group_name: str, rank: int, outdir: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    img_a = _load_gray(str(row["path_a"]))
    img_b = _load_gray(str(row["path_b"]))
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=int(args.target_size), blur_ksize=int(args.blur_ksize)))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=int(args.target_size), blur_ksize=int(args.blur_ksize)))
    meta_a = _parse_filename_metadata(str(row["path_a"]))
    meta_b = _parse_filename_metadata(str(row["path_b"]))
    subject_a = str(row.get("subject_a", meta_a["subject"]))
    subject_b = str(row.get("subject_b", meta_b["subject"]))
    frgp = str(row.get("frgp", meta_a["frgp"] or meta_b["frgp"]))
    header = np.full((76, 1500, 3), 32, dtype=np.uint8)
    title = (
        f"{group_name} #{rank} split={row.get('split', '')} label={row.get('label', '')} "
        f"subject={subject_a}->{subject_b} frgp={frgp} score={float(row.get('score', 0.0)):.6g} "
        f"matches={int(row.get('matches', 0))} inliers={int(row.get('inliers', 0))} "
        f"inlier_ratio={float(row.get('inlier_ratio', 0.0)):.3f}"
    )
    cv2.putText(header, title[:170], (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        "preprocess target_size=768 blur_ksize=0; Lowe ratio=0.75; RANSAC=affine_full_2d threshold=3.0",
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    rows = [
        header,
        _side_by_side(img_a, img_b, "raw plain", "raw roll", height=330),
        _side_by_side(proc_a, proc_b, "preprocessed plain", "preprocessed roll", height=330),
    ]
    keypoints, good_matches, inliers, diagnostics = _sift_v2_views(
        img_a,
        img_b,
        target_size=int(args.target_size),
        nfeatures=int(args.nfeatures),
        blur_ksize=int(args.blur_ksize),
        ratio=float(args.ratio),
        ransac_thresh=float(args.ransac_thresh),
    )
    rows.extend([keypoints, good_matches, inliers])
    width = max(item.shape[1] for item in rows)
    padded = [_pad_to_width(item, width) for item in rows]
    gutter = np.full((12, width, 3), 245, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for item in padded:
        if stacked:
            stacked.append(gutter)
        stacked.append(item)
    sheet = np.vstack(stacked)
    path = outdir / f"{_case_slug(row, group_name, rank)}.png"
    cv2.imwrite(str(path), sheet)
    return path, {
        "group": group_name,
        "rank": int(rank),
        "sheet": str(path),
        "label": int(row.get("label", 0)),
        "split": str(row.get("split", "")),
        "subject_a": subject_a,
        "subject_b": subject_b,
        "frgp": frgp,
        "path_a": str(row["path_a"]),
        "path_b": str(row["path_b"]),
        "score": float(row.get("score", 0.0)),
        "matches": int(row.get("matches", 0)),
        "inliers": int(row.get("inliers", 0)),
        "inlier_ratio": float(row.get("inlier_ratio", 0.0)),
        **diagnostics,
    }


def _selected_cases(scores: pd.DataFrame, threshold: float, top_n: int) -> dict[str, pd.DataFrame]:
    split = scores["split"].astype(str).str.strip().str.lower()
    labels = scores["label"].astype(int)
    accepted = scores["score"].astype(float) >= float(threshold)
    test = split == "test"
    return {
        "top_accepted_true_positives": (
            scores.loc[test & (labels == 1) & accepted].sort_values("score", ascending=False).head(int(top_n))
        ),
        "top_rejected_false_negatives": (
            scores.loc[test & (labels == 1) & (~accepted)].sort_values("score", ascending=False).head(int(top_n))
        ),
        "all_false_accepts": scores.loc[test & (labels == 0) & accepted].sort_values("score", ascending=False),
    }


def generate_visual_audit(
    score_dir: str | Path = DEFAULT_SCORE_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    target_far: float = 0.01,
    top_n: int = 12,
    target_size: int = 768,
    nfeatures: int = 3000,
    blur_ksize: int = 0,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
) -> dict[str, Path]:
    scores_path = parse_file_uri(score_dir)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    scores = _load_scores(scores_path)
    threshold, calibration_false_accepts, calibration_far = _calibrate_threshold(scores, float(target_far))
    groups = _selected_cases(scores, threshold, int(top_n))
    args = argparse.Namespace(
        target_size=int(target_size),
        nfeatures=int(nfeatures),
        blur_ksize=int(blur_ksize),
        ratio=float(ratio),
        ransac_thresh=float(ransac_thresh),
    )
    rows: list[dict[str, Any]] = []
    md_lines = [
        "# SIFT Plain/Roll v2 Visual Audit",
        "",
        f"SIFT v2 score folder: `{scores_path}`",
        f"Threshold calibrated on original validation negatives at target FAR {float(target_far):.1%}: `{threshold:.6g}`",
        f"Validation calibration false accepts: {int(calibration_false_accepts)}; val FAR: {float(calibration_far):.6g}",
        "",
        "| group | rank | split | label | subject/frgp | score | matches | inliers | inlier ratio | sheet |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    created: dict[str, Path] = {}
    for group_name, group_df in groups.items():
        if group_df.empty:
            md_lines.append(f"| {group_name} | 0 |  |  | none |  |  |  |  |  |")
            continue
        for rank, (_, row) in enumerate(group_df.iterrows(), start=1):
            sheet_path, payload = _make_case_sheet(row, group_name, rank, output, args)
            rows.append(payload)
            created[f"{group_name}_{rank:02d}"] = sheet_path
            rel = sheet_path.name
            subject = f"{payload['subject_a']}->{payload['subject_b']} / {payload['frgp']}"
            md_lines.append(
                f"| {group_name} | {rank} | {payload['split']} | {payload['label']} | {subject} | "
                f"{payload['score']:.6g} | {payload['matches']} | {payload['inliers']} | "
                f"{payload['inlier_ratio']:.3f} | [{rel}]({rel}) |"
            )
    index_csv = output / "cases.csv"
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    index_md = output / "index.md"
    index_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    created["cases_csv"] = index_csv
    created["index"] = index_md
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SIFT Plain/Roll v2 visual audit sheets.")
    parser.add_argument("--score_dir", default=str(DEFAULT_SCORE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--target_far", type=float, default=0.01)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--target_size", type=int, default=768)
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--blur_ksize", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_visual_audit(
        args.score_dir,
        args.outdir,
        target_far=float(args.target_far),
        top_n=int(args.top_n),
        target_size=int(args.target_size),
        nfeatures=int(args.nfeatures),
        blur_ksize=int(args.blur_ksize),
        ratio=float(args.ratio),
        ransac_thresh=float(args.ransac_thresh),
    )
    print("Wrote SIFT Plain/Roll v2 visual audit:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
