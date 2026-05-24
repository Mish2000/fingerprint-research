from __future__ import annotations

import argparse
import hashlib
import math
import os
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

from src.fpbench.matchers.minutiae_matcher import (  # noqa: E402
    MinutiaeConfig,
    _as_gray_u8,
    _make_roi_mask,
    _preprocess_for_minutiae,
    _select_extraction_profile,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, preprocess_image  # noqa: E402


DEFAULT_DIAG_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
DEFAULT_OUTDIR = DEFAULT_DIAG_DIR / "visual_audit"
INPUT_FILES = (
    "top_positive_failures.csv",
    "top_positive_successes.csv",
    "top_negative_false_accepts.csv",
)


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


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, height, 3), dtype=np.uint8)
    scale = float(height) / float(h)
    width = max(1, int(round(w * scale)))
    return cv2.resize(_to_bgr(img), (width, int(height)), interpolation=cv2.INTER_AREA)


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= int(width):
        return _to_bgr(img)
    scale = float(width) / float(w)
    height = max(1, int(round(h * scale)))
    return cv2.resize(_to_bgr(img), (int(width), height), interpolation=cv2.INTER_AREA)


def _pad_to_width(img: np.ndarray, width: int, value: int = 245) -> np.ndarray:
    out = _to_bgr(img)
    if out.shape[1] >= width:
        return out
    pad = np.full((out.shape[0], width - out.shape[1], 3), int(value), dtype=np.uint8)
    return np.hstack([out, pad])


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = _to_bgr(img)
    bar = np.full((34, out.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text[:150], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def _side_by_side(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str, *, height: int = 300) -> np.ndarray:
    left_img = _label(_resize_to_height(left, height), left_label)
    right_img = _label(_resize_to_height(right, height), right_label)
    max_h = max(left_img.shape[0], right_img.shape[0])
    left_img = np.pad(left_img, ((0, max_h - left_img.shape[0]), (0, 0), (0, 0)), constant_values=245)
    right_img = np.pad(right_img, ((0, max_h - right_img.shape[0]), (0, 0), (0, 0)), constant_values=245)
    gutter = np.full((max_h, 12, 3), 245, dtype=np.uint8)
    return np.hstack([left_img, gutter, right_img])


def _sift_views(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    target_size: int,
    nfeatures: int,
    ratio: float,
    ransac_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=int(target_size)))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=int(target_size)))
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    kps_a, desc_a = sift.detectAndCompute(proc_a, None)
    kps_b, desc_b = sift.detectAndCompute(proc_b, None)
    kps_a = kps_a or []
    kps_b = kps_b or []
    key_a = cv2.drawKeypoints(proc_a, kps_a, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    key_b = cv2.drawKeypoints(proc_b, kps_b, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints = _side_by_side(key_a, key_b, f"SIFT keypoints plain: {len(kps_a)}", f"SIFT keypoints roll: {len(kps_b)}", height=320)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        blank = np.full((240, 900, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "No SIFT descriptors", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        return keypoints, _label(blank, "SIFT good matches"), _label(blank, "SIFT inliers")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for item in knn:
        if len(item) == 2:
            first, second = item
            if first.distance < float(ratio) * second.distance:
                good.append(first)
    good_sorted = sorted(good, key=lambda m: float(m.distance))
    draw_good = good_sorted[:80]
    good_img = cv2.drawMatches(
        proc_a,
        kps_a,
        proc_b,
        kps_b,
        draw_good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    good_img = _label(_fit_width(good_img, 1500), f"SIFT Lowe-ratio good matches: {len(good)}")

    inlier_img = np.full_like(good_img, 245)
    if len(good) >= 8:
        pts_a = np.float32([kps_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kps_b[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=float(ransac_thresh))
        if mask is not None:
            inlier_flags = mask.ravel().astype(bool)
            inlier_matches = [m for m, keep in zip(good, inlier_flags) if keep]
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
            inlier_img = _label(_fit_width(raw, 1500), f"SIFT homography inliers: {int(np.sum(inlier_flags))}")
    else:
        blank = np.full((240, 900, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "Too few good matches for homography", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        inlier_img = _label(blank, "SIFT homography inliers")
    return keypoints, good_img, inlier_img


def _has_dense_signal(row: pd.Series) -> bool:
    for key in ("dense_skeleton", "extraction_quality_flags_a", "extraction_quality_flags_b"):
        value = row.get(key, "")
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).lower()
        if text in {"true", "1"} or "dense_skeleton" in text:
            return True
    return False


def _minutiae_overlay_one(img: np.ndarray, title: str, cfg: MinutiaeConfig) -> np.ndarray:
    gray = _preprocess_for_minutiae(_as_gray_u8(img), cfg)
    roi, warnings = _make_roi_mask(gray, None, cfg)
    profile = _select_extraction_profile(gray, roi, cfg)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    skeleton = profile.skeleton > 0
    out[skeleton] = (255, 80, 30)
    contours, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 220), 1)
    for point in profile.extraction.points:
        color = (30, 210, 30) if point.kind == "ending" else (30, 30, 230)
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(out, center, 4, color, 1, cv2.LINE_AA)
        tip = (
            int(round(point.x + 10.0 * math.cos(point.theta))),
            int(round(point.y + 10.0 * math.sin(point.theta))),
        )
        cv2.line(out, center, tip, color, 1, cv2.LINE_AA)
    flags = ";".join(profile.quality_flags) or "none"
    warn = ";".join(warnings) or "none"
    label = (
        f"{title} minutiae skeleton: points={len(profile.extraction.points)} "
        f"density={profile.skeleton_density:.3f} flags={flags} warnings={warn}"
    )
    return _label(_resize_to_height(out, 340), label)


def _minutiae_overlay_pair(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    cfg = MinutiaeConfig()
    left = _minutiae_overlay_one(img_a, "plain", cfg)
    right = _minutiae_overlay_one(img_b, "roll", cfg)
    max_h = max(left.shape[0], right.shape[0])
    left = np.pad(left, ((0, max_h - left.shape[0]), (0, 0), (0, 0)), constant_values=245)
    right = np.pad(right, ((0, max_h - right.shape[0]), (0, 0), (0, 0)), constant_values=245)
    return np.hstack([left, np.full((max_h, 12, 3), 245, dtype=np.uint8), right])


def _case_slug(row: pd.Series, group_name: str, index: int) -> str:
    payload = f"{row.get('path_a', '')}|{row.get('path_b', '')}|{group_name}|{index}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    method = str(row.get("method", "case")).replace(" ", "_")
    return f"{group_name}_{index:02d}_{method}_{digest}"


def _make_case_sheet(row: pd.Series, group_name: str, index: int, outdir: Path, args: argparse.Namespace) -> Path:
    img_a = _load_gray(str(row["path_a"]))
    img_b = _load_gray(str(row["path_b"]))
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=int(args.target_size)))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=int(args.target_size)))
    header = np.full((58, 1500, 3), 32, dtype=np.uint8)
    title = (
        f"{group_name} #{index} method={row.get('method', '')} label={row.get('label', '')} "
        f"score={row.get('score', '')} reason={row.get('failure_reason', '')}"
    )
    cv2.putText(header, title[:160], (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    rows = [
        header,
        _side_by_side(img_a, img_b, "raw plain", "raw roll", height=330),
        _side_by_side(proc_a, proc_b, "preprocessed plain", "preprocessed roll", height=330),
    ]
    keypoints, good_matches, inliers = _sift_views(
        img_a,
        img_b,
        target_size=int(args.target_size),
        nfeatures=int(args.nfeatures),
        ratio=float(args.ratio),
        ransac_thresh=float(args.ransac_thresh),
    )
    rows.extend([keypoints, good_matches, inliers])
    if _has_dense_signal(row):
        rows.append(_minutiae_overlay_pair(img_a, img_b))
    width = max(item.shape[1] for item in rows)
    padded = [_pad_to_width(item, width) for item in rows]
    gutter = np.full((12, width, 3), 245, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for item in padded:
        if stacked:
            stacked.append(gutter)
        stacked.append(item)
    sheet = np.vstack(stacked)
    path = outdir / f"{_case_slug(row, group_name, index)}.png"
    cv2.imwrite(str(path), sheet)
    return path


def generate_visual_audit(
    diag_dir: str | Path = DEFAULT_DIAG_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    limit_per_input: int = 8,
    target_size: int = 512,
    nfeatures: int = 1500,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
) -> dict[str, Path]:
    diag = parse_file_uri(diag_dir)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        target_size=int(target_size),
        nfeatures=int(nfeatures),
        ratio=float(ratio),
        ransac_thresh=float(ransac_thresh),
    )
    rows_md = [
        "# Plain-vs-Roll Visual Audit",
        "",
        f"Source diagnostics folder: `{diag}`",
        "",
        "| group | rank | method | label | score | failure reason | sheet |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    created: dict[str, Path] = {}
    for input_name in INPUT_FILES:
        input_path = diag / input_name
        if not input_path.exists():
            continue
        df = pd.read_csv(input_path)
        group_name = input_path.stem
        for rank, (_, row) in enumerate(df.head(int(limit_per_input)).iterrows(), start=1):
            sheet_path = _make_case_sheet(row, group_name, rank, output, args)
            created[f"{group_name}_{rank:02d}"] = sheet_path
            rel = sheet_path.name
            score = pd.to_numeric(pd.Series([row.get("score", np.nan)]), errors="coerce").iloc[0]
            rows_md.append(
                f"| {group_name} | {rank} | {row.get('method', '')} | {row.get('label', '')} | "
                f"{float(score):.6g} | {row.get('failure_reason', '')} | [{rel}]({rel}) |"
            )
    index_path = output / "index.md"
    index_path.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    created["index"] = index_path
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate visual audit sheets for plain-vs-roll diagnostics.")
    parser.add_argument("--diag_dir", default=str(DEFAULT_DIAG_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--limit_per_input", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--nfeatures", type=int, default=1500)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_visual_audit(
        args.diag_dir,
        args.outdir,
        limit_per_input=int(args.limit_per_input),
        target_size=int(args.target_size),
        nfeatures=int(args.nfeatures),
        ratio=float(args.ratio),
        ransac_thresh=float(args.ransac_thresh),
    )
    print("Wrote visual audit:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
