import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dedicated_matcher import DedicatedMatcher, draw_match_viz, _norm_path
from src.preprocess import load_gray, preprocess_image, PreprocessConfig


def load_capture_map(manifest_csv: Path) -> dict:
    df = pd.read_csv(manifest_csv)
    # store normalized path -> capture
    m = {}
    for _, r in df.iterrows():
        p = str(r["path"])
        cap = str(r["capture"])
        m[_norm_path(p)] = cap
    return m


def infer_capture_from_name(path: str) -> str | None:
    name = Path(path).name.lower()
    if "_plain_" in name:
        return "plain"
    if "_roll_" in name:
        return "roll"
    return None

def resolve_capture(path: str, cap_map: dict, fallback: str | None) -> str:
    key = _norm_path(path)
    if key in cap_map:
        return str(cap_map[key])
    if fallback is not None:
        return str(fallback)

    inferred = infer_capture_from_name(path)
    if inferred is not None:
        return inferred

    raise ValueError(
        f"Could not find capture type for:\n  {path}\n"
        f"Either:\n"
        f"  1) pass --capture_a/--capture_b manually, or\n"
        f"  2) ensure this exact path exists in manifest.csv, or\n"
        f"  3) include '_plain_' or '_roll_' in the filename."
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_a", required=True, type=str)
    ap.add_argument("--img_b", required=True, type=str)

    ap.add_argument("--manifest", type=str, default="data/processed/nist_sd300b/manifest.csv")
    ap.add_argument("--capture_a", type=str, default=None, choices=[None, "plain", "roll"])
    ap.add_argument("--capture_b", type=str, default=None, choices=[None, "plain", "roll"])

    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="reports/week08/smoke")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_map = load_capture_map(Path(args.manifest))

    cap_a = resolve_capture(args.img_a, cap_map, args.capture_a)
    cap_b = resolve_capture(args.img_b, cap_map, args.capture_b)

    matcher = DedicatedMatcher(ckpt_path=args.ckpt)

    res = matcher.score_pair(args.img_a, args.img_b, capture_a=cap_a, capture_b=cap_b)

    # Build a visualization
    cfg = PreprocessConfig(target_size=512, clahe_clip=2.0, clahe_grid=(8, 8), blur_ksize=3)
    imgA = preprocess_image(load_gray(args.img_a), cfg)
    imgB = preprocess_image(load_gray(args.img_b), cfg)

    viz = draw_match_viz(imgA, imgB, res.matches, max_draw=120)
    out_png = out_dir / "match_viz.png"
    cv2.imwrite(str(out_png), viz)

    # Save JSON for inspection
    out_json = out_dir / "match_result.json"
    out_json.write_text(
        __import__("json").dumps(
            {
                "img_a": args.img_a,
                "img_b": args.img_b,
                "capture_a": cap_a,
                "capture_b": cap_b,
                "score": res.score,
                "inliers_count": res.inliers_count,
                "tentative_count": res.tentative_count,
                "latency_ms": res.latency_ms,
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n[DEDICATED MATCHER SMOKE]")
    print(f"score={res.score:.4f}  inliers={res.inliers_count}/{res.tentative_count}")
    print("latency_ms:", res.latency_ms)
    print("Wrote:", out_png)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()
