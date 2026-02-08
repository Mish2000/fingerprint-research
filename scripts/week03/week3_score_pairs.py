from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(r"/")
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import PreprocessConfig, load_gray, preprocess_image, fingerprint_roi_mask
from src.matching_baseline import ORBConfig, score_pair

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "nist_sd300b"
OUT_DIR = PROJECT_ROOT / "reports" / "week03"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rectangular_gate_mask(shape_hw, capture: str, top_plain: float = 0.18, top_roll: float = 0.05, border: int = 12):
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    top = int(round((top_plain if capture == "plain" else top_roll) * h))
    y0, y1 = top, h - border
    x0, x1 = border, w - border
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = 255
    return m


def build_final_mask(img_u8: np.ndarray, capture: str) -> np.ndarray:
    mask_roi = fingerprint_roi_mask(img_u8)
    mask_gate = rectangular_gate_mask(img_u8.shape[:2], capture=capture)

    roi_white = float((mask_roi > 0).mean())
    roi_bad = (roi_white > 0.70) or (roi_white < 0.10)

    if roi_bad:
        return mask_gate
    mask_final = cv2.bitwise_and(mask_roi, mask_gate)
    if float((mask_final > 0).mean()) < 0.12:
        return mask_gate
    return mask_final


def run_scoring(split: str = "test", max_pairs: int | None = 2000):
    pos = pd.read_csv(DATASET_DIR / "pairs_pos.csv")
    neg = pd.read_csv(DATASET_DIR / "pairs_neg.csv")

    pos = pos[pos["split"] == split].copy()
    neg = neg[neg["split"] == split].copy()

    if max_pairs is not None:
        pos = pos.head(max_pairs)
        neg = neg.head(max_pairs)

    cfg = PreprocessConfig(target_size=512, clahe_clip=2.0, blur_ksize=3)
    orb_cfg = ORBConfig(nfeatures=1500, fastThreshold=10)

    rows = []

    def process_df(df, label: int):
        for r in df.itertuples(index=False):
            img_a = preprocess_image(load_gray(r.path_a), cfg)
            img_b = preprocess_image(load_gray(r.path_b), cfg)

            # infer capture from filename (plain/roll)
            cap_a = "plain" if "_plain_" in Path(r.path_a).name else "roll"
            cap_b = "plain" if "_plain_" in Path(r.path_b).name else "roll"

            mask_a = build_final_mask(img_a, cap_a)
            mask_b = build_final_mask(img_b, cap_b)

            out = score_pair(img_a, img_b, mask_a, mask_b, orb_cfg, ratio=0.75, reproj=3.0)
            rows.append({
                "label": label,
                "split": split,
                "path_a": r.path_a,
                "path_b": r.path_b,
                **out
            })

    process_df(pos, 1)
    process_df(neg, 0)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"scores_{split}.csv", index=False)

    y = res["label"].values
    s = res["score"].values
    auc = roc_auc_score(y, s)

    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    print(f"Split={split} | N={len(res)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print(f"Saved: {OUT_DIR / f'scores_{split}.csv'}")


if __name__ == "__main__":
    run_scoring(split="test", max_pairs=2000)
