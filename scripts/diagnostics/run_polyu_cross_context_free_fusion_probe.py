"""Context-free, low-capacity PolyU Cross statistical fusion probe (Phase 4A.2B-2).

Tests whether the *fusion hypothesis itself* is useful when a NEW, context-free,
PolyU-consistent statistical model is trained correctly - as opposed to the
frozen SD300 fusion models, which 4A.2B-1 showed are schema-incompatible.

Protocol (strict):
* Context-free features only: matcher scores + deterministic image-quality
  features + a small predeclared interaction set. NEVER dataset/session/subject/
  finger_unit/pair_id/path/frgp/modality as predictive features.
* Model: StandardScaler + L2 LogisticRegression (C=1.0, class_weight=balanced),
  fixed random_state. Ranking by decision_function. No grid search.
* Subject-disjoint 5-fold CV *inside TRAIN* for feature-group comparison and
  selection (NOT naive GroupKFold on subject_a): each fold partitions TRAIN
  finger_units into 5 disjoint groups; fold-train/val pairs require BOTH pair
  identities inside the training-union / held-out partition respectively.
* Controls: quality-only (Q0) and deterministic shuffled-label.
* After freezing the selected group on TRAIN CV: fit once on full TRAIN and
  evaluate VAL exactly once. TEST is never read.

Nothing here refits the deep model or SourceAFIS, modifies canonical pairs/
manifests, or touches TEST.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.datasets.polyu_cross_pairs import DATASET as POLYU_CROSS_DATASET, load_polyu_cross_pairs
from src.fpbench.universal.quality import extract_image_quality
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_context_free_fusion_probe_v0"
SOURCEAFIS_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0_sourceafis_real"
SIFT_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0"
DEEP_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_deep_v0"
RUN_SCHEMA_VERSION = "polyu_cross_context_free_fusion_probe_v0"

K_FOLDS = 5
RANDOM_STATE = 13
LR_C = 1.0
SHUFFLE_SEED = 20260704
SELECT_TOL = 0.01  # prefer simpler group if within this mean-AUC of the best
MAJORITY_FOLDS = 3  # improvement over G0 required on >= this many folds

# Quality bases (aspect_ratio excluded: nearly modality-identifying here).
QUALITY_BASES = ["std_intensity", "contrast_proxy", "foreground_ratio", "sharpness_laplacian_var", "edge_density"]
QUALITY_FEATURES = [f"{side}_{base}" for base in QUALITY_BASES for side in ("a", "b")]
QUALITY_DELTAS = [f"pair_{base}_abs_delta" for base in QUALITY_BASES]

INTERACTIONS = {
    "int_p2_x_a_contrast": ("p2_sourceafis_score", "a_contrast_proxy"),
    "int_p2_x_a_sharpness": ("p2_sourceafis_score", "a_sharpness_laplacian_var"),
    "int_p2_x_a_foreground": ("p2_sourceafis_score", "a_foreground_ratio"),
    "int_deep_x_a_contrast": ("deep_logit", "a_contrast_proxy"),
    "int_deep_x_a_sharpness": ("deep_logit", "a_sharpness_laplacian_var"),
}

G0 = ["p2_sourceafis_score"]
G1 = G0 + ["deep_logit"]
G2 = G1 + ["sift_score", "sift_inliers", "sift_matches", "sift_k1", "sift_k2"]
G3 = G2 + QUALITY_FEATURES + QUALITY_DELTAS
G4 = G3 + list(INTERACTIONS.keys())

FEATURE_GROUPS: dict[str, list[str]] = {
    "Q0_quality_only": QUALITY_FEATURES + QUALITY_DELTAS,
    "G0_p2_sourceafis": G0,
    "G1_p2_plus_deep": G1,
    "G2_matcher_geometry": G2,
    "G3_quality_aware": G3,
    "G4_reliability_interactions": G4,
}
FUSION_ORDER = ["G0_p2_sourceafis", "G1_p2_plus_deep", "G2_matcher_geometry", "G3_quality_aware", "G4_reliability_interactions"]


class ProbeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=LR_C, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE, solver="lbfgs")),
    ])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2 or not np.isfinite(scores).all():
        finite = np.isfinite(scores)
        labels, scores = labels[finite], scores[finite]
        if labels.size == 0 or np.unique(labels).size < 2:
            return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def _read_join(path: Path, cols: dict[str, str], *, keep_paths: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"pair_id": df["pair_id"].astype(str), "label": df["label"].astype(int)})
    for src, dst in cols.items():
        if src not in df.columns:
            raise ProbeError(f"{path.name} missing required column {src!r}")
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    if keep_paths:
        for p in ("resolved_path_a", "resolved_path_b"):
            out[p] = df[p].astype(str)
    return out


def build_feature_table(split: str, manifest_dir: Path, p2_dir: Path) -> pd.DataFrame:
    pairs = load_polyu_cross_pairs(manifest_dir / f"pairs_{split}.csv")
    base = pd.DataFrame({
        "pair_id": pairs["pair_id"].astype(str),
        "label": pairs["label"].astype(int),
        "finger_unit_a": pairs["finger_unit_a"].astype(str),
        "finger_unit_b": pairs["finger_unit_b"].astype(str),
    })

    sa = _read_join(REPO_ROOT / SOURCEAFIS_DIR / f"scores_polyu_cross_sourceafis_open_{split}.csv",
                    {"score": "raw_sourceafis_score"}, keep_paths=True)
    p2 = _read_join(p2_dir / f"p2_sourceafis_{split}.csv", {"score": "p2_sourceafis_score"})
    sift = _read_join(REPO_ROOT / SIFT_DIR / f"scores_polyu_cross_sift_{split}.csv",
                      {"score": "sift_score", "inliers": "sift_inliers", "matches": "sift_matches", "k1": "sift_k1", "k2": "sift_k2"})
    deep = _read_join(REPO_ROOT / DEEP_DIR / f"scores_polyu_cross_deep_pair_reranker_{split}.csv",
                      {"logit": "deep_logit"})

    n0 = len(base)
    t = base
    for name, frame in (("raw_sourceafis", sa), ("p2_sourceafis", p2), ("sift", sift), ("deep", deep)):
        if len(frame) != n0:
            raise ProbeError(f"{name} score rows ({len(frame)}) != pairs ({n0}) for split={split}")
        # exact one-to-one pair_id join; refuse silent imputation
        merged = t.merge(frame, on=["pair_id", "label"], how="left", validate="one_to_one")
        newcols = [c for c in frame.columns if c not in ("pair_id", "label")]
        if merged[newcols].isna().any().any():
            miss = int(merged[newcols].isna().any(axis=1).sum())
            raise ProbeError(f"{name}: {miss} pair(s) failed exact pair_id join for split={split} (no imputation allowed)")
        t = merged

    # Quality on RAW images (a=contactless probe, b=contact gallery).
    qbases = ["std_intensity", "contrast_proxy", "foreground_ratio", "sharpness_laplacian_var", "edge_density"]
    paths = pd.unique(pd.concat([t["resolved_path_a"], t["resolved_path_b"]], ignore_index=True))
    qmap = {p: extract_image_quality(p, repo_root=REPO_ROOT) for p in paths}
    for side, col in (("a", "resolved_path_a"), ("b", "resolved_path_b")):
        for b in qbases:
            t[f"{side}_{b}"] = [float(qmap[p][b]) for p in t[col]]
    for b in qbases:
        t[f"pair_{b}_abs_delta"] = (pd.to_numeric(t[f"a_{b}"], errors="coerce") - pd.to_numeric(t[f"b_{b}"], errors="coerce")).abs()

    for name, (x, y) in INTERACTIONS.items():
        t[name] = pd.to_numeric(t[x], errors="coerce") * pd.to_numeric(t[y], errors="coerce")

    t["split"] = split
    # Validate all model-usable features are finite.
    all_feats = sorted(set(sum(FEATURE_GROUPS.values(), [])) | {"raw_sourceafis_score"})
    bad = [c for c in all_feats if c in t.columns and not np.isfinite(pd.to_numeric(t[c], errors="coerce")).all()]
    if bad:
        raise ProbeError(f"Non-finite feature values in {bad} for split={split}")
    return t.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subject-disjoint CV
# ---------------------------------------------------------------------------
def assign_folds(finger_units: list[str], k: int = K_FOLDS) -> dict[str, int]:
    """Deterministic round-robin partition of sorted finger_units into k groups."""
    ordered = sorted(finger_units, key=lambda x: (len(str(x)), str(x)))
    return {fu: (i % k) for i, fu in enumerate(ordered)}


def fold_masks(table: pd.DataFrame, fold_of: dict[str, int], held: int):
    fa = table["finger_unit_a"].astype(str).map(fold_of)
    fb = table["finger_unit_b"].astype(str).map(fold_of)
    val_mask = (fa == held) & (fb == held)
    train_mask = (fa != held) & (fb != held)
    return train_mask.to_numpy(), val_mask.to_numpy()


def cv_evaluate(table: pd.DataFrame, fold_of: dict[str, int], groups: dict[str, list[str]], *, labels_override: Optional[np.ndarray] = None) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    labels_all = np.asarray(labels_override if labels_override is not None else table["label"].astype(int).to_numpy())
    rows: list[dict[str, Any]] = []
    coefs: dict[str, list[dict]] = {g: [] for g in groups}
    for group, feats in groups.items():
        X = table[feats].to_numpy(dtype=float)
        for held in range(K_FOLDS):
            tr_mask, va_mask = fold_masks(table, fold_of, held)
            ytr, yva = labels_all[tr_mask], labels_all[va_mask]
            if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
                rows.append({"group": group, "fold": held, "auc": float("nan"), "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
                             "n_val_pos": int((yva == 1).sum()), "n_val_neg": int((yva == 0).sum()), "converged": False})
                continue
            model = make_model()
            model.fit(X[tr_mask], ytr)
            dscore = model.decision_function(X[va_mask])
            auc = _auc(yva, dscore)
            lr = model.named_steps["lr"]
            n_iter = int(np.max(getattr(lr, "n_iter_", [0])))
            rows.append({"group": group, "fold": held, "auc": auc, "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
                         "n_val_pos": int((yva == 1).sum()), "n_val_neg": int((yva == 0).sum()),
                         "converged": bool(n_iter < 5000), "n_iter": n_iter})
            coefs[group].append({"fold": held, **{f: float(c) for f, c in zip(feats, lr.coef_[0])}})
    return pd.DataFrame(rows), coefs


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    g0 = cv[cv["group"] == "G0_p2_sourceafis"].set_index("fold")["auc"]
    rows = []
    for group, sub in cv.groupby("group", sort=False):
        aucs = pd.to_numeric(sub["auc"], errors="coerce")
        by_fold = sub.set_index("fold")["auc"]
        beats = int(sum(1 for f in range(K_FOLDS) if np.isfinite(by_fold.get(f, np.nan)) and np.isfinite(g0.get(f, np.nan)) and by_fold[f] > g0[f]))
        rows.append({"group": group, "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std(ddof=1)),
                     "min_auc": float(aucs.min()), "max_auc": float(aucs.max()), "n_folds": int(aucs.notna().sum()),
                     "n_folds_beating_G0": beats, "all_converged": bool(sub["converged"].all())})
    order = {g: i for i, g in enumerate(FEATURE_GROUPS)}
    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.map(order)).reset_index(drop=True)


def select_group(summary: pd.DataFrame) -> dict[str, Any]:
    fus = summary[summary["group"].isin(FUSION_ORDER)].set_index("group")
    best_mean = float(fus["mean_auc"].max())
    # Simplest group within SELECT_TOL of the best.
    selected = None
    for g in FUSION_ORDER:
        if float(fus.loc[g, "mean_auc"]) >= best_mean - SELECT_TOL:
            selected = g
            break
    # Improvement-over-G0 requirement.
    fell_back = False
    if selected != "G0_p2_sourceafis":
        if int(fus.loc[selected, "n_folds_beating_G0"]) < MAJORITY_FOLDS:
            selected = "G0_p2_sourceafis"
            fell_back = True
    return {"selected_group": selected, "best_mean_auc": best_mean, "fell_back_to_G0": fell_back,
            "rule": f"simplest group within {SELECT_TOL} of best mean CV AUC; if non-G0, require > G0 on >= {MAJORITY_FOLDS}/{K_FOLDS} folds"}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(*, manifest_dir: Path, outdir: Path, p2_dir: Path) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = build_feature_table("train", manifest_dir, p2_dir)
    val = build_feature_table("val", manifest_dir, p2_dir)

    finger_units = sorted(set(train["finger_unit_a"]) | set(train["finger_unit_b"]))
    fold_of = assign_folds(finger_units)
    (outdir / "cv_fold_assignments.json").write_text(json.dumps({
        "k": K_FOLDS, "method": "round_robin_on_sorted_finger_units",
        "n_finger_units": len(finger_units),
        "partition_sizes": {str(k): int(sum(1 for v in fold_of.values() if v == k)) for k in range(K_FOLDS)},
        "assignments": fold_of,
    }, indent=2), encoding="utf-8")

    # --- CV over all groups (true labels) ---
    cv, coefs = cv_evaluate(train, fold_of, FEATURE_GROUPS)
    cv.to_csv(outdir / "cv_metrics.csv", index=False)
    summary = summarize_cv(cv)
    summary.to_csv(outdir / "cv_summary.csv", index=False)

    # --- Controls ---
    control_rows: list[dict[str, Any]] = []
    q0 = summary[summary["group"] == "Q0_quality_only"].iloc[0]
    control_rows.append({"control": "quality_only_Q0", "mean_cv_auc": float(q0["mean_auc"]), "std_cv_auc": float(q0["std_auc"]), "note": "negative control"})
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled_labels = train["label"].to_numpy().copy()
    rng.shuffle(shuffled_labels)
    shuf_cv, _ = cv_evaluate(train, fold_of, {"G4_reliability_interactions": G4, "G0_p2_sourceafis": G0}, labels_override=shuffled_labels)
    for group, sub in shuf_cv.groupby("group"):
        control_rows.append({"control": f"shuffled_label::{group}", "mean_cv_auc": float(pd.to_numeric(sub["auc"], errors="coerce").mean()),
                             "std_cv_auc": float(pd.to_numeric(sub["auc"], errors="coerce").std(ddof=1)), "note": "deterministic label permutation; expect ~0.50"})
    controls = pd.DataFrame(control_rows)
    controls.to_csv(outdir / "controls.csv", index=False)

    # --- Selection (TRAIN CV only) ---
    selection = select_group(summary)
    selected = selection["selected_group"]

    # --- Coefficient stability for selected group ---
    coef_rows: list[dict[str, Any]] = []
    sel_coefs = pd.DataFrame(coefs[selected])
    for feat in FEATURE_GROUPS[selected]:
        vals = pd.to_numeric(sel_coefs[feat], errors="coerce").to_numpy() if feat in sel_coefs.columns else np.array([])
        mean = float(np.mean(vals)) if vals.size else float("nan")
        coef_rows.append({"group": selected, "feature": feat, "coef_mean": mean,
                          "coef_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                          "sign_consistency": float(np.mean(np.sign(vals) == np.sign(mean))) if vals.size else float("nan")})
    coef_stability = pd.DataFrame(coef_rows)
    coef_stability.to_csv(outdir / "coefficient_stability.csv", index=False)

    # --- ONE-TIME VAL evaluation (selected + G0 baseline) ---
    val_rows: list[dict[str, Any]] = []
    full_coefs: dict[str, dict[str, float]] = {}
    for group in dict.fromkeys([selected, "G0_p2_sourceafis"]):
        feats = FEATURE_GROUPS[group]
        model = make_model()
        model.fit(train[feats].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
        dscore = model.decision_function(val[feats].to_numpy(dtype=float))
        val_rows.append({"group": group, "is_selected": group == selected, "val_auc": _auc(val["label"].to_numpy(), dscore),
                         "n_val": int(len(val)), "n_val_pos": int((val["label"] == 1).sum()), "n_val_neg": int((val["label"] == 0).sum())})
        full_coefs[group] = {f: float(c) for f, c in zip(feats, model.named_steps["lr"].coef_[0])}
    val_result = pd.DataFrame(val_rows)
    val_result.to_csv(outdir / "val_result.csv", index=False)

    # --- Classification ---
    sel_row = summary[summary["group"] == selected].iloc[0]
    g0_row = summary[summary["group"] == "G0_p2_sourceafis"].iloc[0]
    sel_val = float(val_result[val_result["group"] == selected].iloc[0]["val_auc"])
    g0_val = float(val_result[val_result["group"] == "G0_p2_sourceafis"].iloc[0]["val_auc"])
    q0_mean = float(q0["mean_auc"])
    shuf_mean = float(np.nanmean([r["mean_cv_auc"] for r in control_rows if r["control"].startswith("shuffled")]))

    cv_gain = float(sel_row["mean_auc"]) - float(g0_row["mean_auc"])
    beats_majority = selected == "G0_p2_sourceafis" or int(sel_row["n_folds_beating_G0"]) >= MAJORITY_FOLDS
    val_survives = sel_val >= g0_val + 0.005
    q0_strong = q0_mean > 0.60
    shuffled_biased = abs(shuf_mean - 0.5) > 0.05

    if q0_strong or shuffled_biased:
        classification = "D. SUSPECTED_SHORTCUT"
    elif selected == "G0_p2_sourceafis" or cv_gain <= SELECT_TOL or not beats_majority:
        classification = "C. NO_COMPLEMENTARITY"
    elif not val_survives:
        classification = "B. TRAIN_ONLY_OVERFIT"
    else:
        classification = "A. STABLE_FUSION_COMPLEMENTARITY"

    # --- Feature schema + manifests ---
    (outdir / "feature_schema.json").write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION,
        "feature_groups": FEATURE_GROUPS, "interactions": {k: list(v) for k, v in INTERACTIONS.items()},
        "excluded_context_features": ["dataset", "session_id", "subject_id", "finger_unit_id", "pair_id", "path_a", "path_b", "frgp", "finger_position", "modality_a", "modality_b", "aspect_ratio"],
        "model": {"pipeline": "StandardScaler -> LogisticRegression", "C": LR_C, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_STATE, "ranking": "decision_function"},
    }, indent=2), encoding="utf-8")

    (outdir / "feature_table_manifest.json").write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION,
        "train_rows": int(len(train)), "val_rows": int(len(val)),
        "score_sources": {
            "raw_sourceafis": SOURCEAFIS_DIR, "p2_sourceafis": str(p2_dir), "sift": SIFT_DIR, "deep": DEEP_DIR,
        },
        "score_shas": {
            "sourceafis_train": sha256_file(REPO_ROOT / SOURCEAFIS_DIR / "scores_polyu_cross_sourceafis_open_train.csv"),
            "p2_train": sha256_file(p2_dir / "p2_sourceafis_train.csv"),
            "deep_train": sha256_file(REPO_ROOT / DEEP_DIR / "scores_polyu_cross_deep_pair_reranker_train.csv"),
            "sift_train": sha256_file(REPO_ROOT / SIFT_DIR / "scores_polyu_cross_sift_train.csv"),
        },
        "join": "exact one-to-one by pair_id; no missing-score imputation", "quality_on": "raw images (a=contactless probe, b=contact gallery)",
    }, indent=2, default=str), encoding="utf-8")

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION, "created_at": utc_now(), "dataset": DATASET_NAME,
        "protocol": "TRAIN subject-disjoint 5-fold CV for selection; VAL evaluated once; TEST never read.",
        "cv": {"k": K_FOLDS, "method": "subject-disjoint (both pair identities inside partition)", "random_state": RANDOM_STATE},
        "selection": selection, "classification": classification,
        "cv_selected": {"group": selected, "mean_auc": float(sel_row["mean_auc"]), "std_auc": float(sel_row["std_auc"]),
                         "min_auc": float(sel_row["min_auc"]), "max_auc": float(sel_row["max_auc"]), "n_folds_beating_G0": int(sel_row["n_folds_beating_G0"])},
        "cv_G0": {"mean_auc": float(g0_row["mean_auc"]), "std_auc": float(g0_row["std_auc"])},
        "cv_gain_over_G0": cv_gain,
        "val": {"selected_val_auc": sel_val, "G0_val_auc": g0_val, "improvement_survives_val": bool(val_survives)},
        "controls": {"quality_only_Q0_mean_cv_auc": q0_mean, "shuffled_label_mean_cv_auc": shuf_mean},
        "selected_full_train_coefficients": full_coefs.get(selected, {}),
        "git": git_info(), "python": {"version": sys.version, "executable": sys.executable}, "platform": platform.platform(),
        "constraints": {"used_test": False, "used_context_features": False, "refit_deep_model": False, "modified_sourceafis": False,
                        "modified_manifest_or_pairs": False, "tree_ensembles_or_nn": False, "grid_search": False, "repeated_val_selection": False},
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {"cv": cv, "summary": summary, "controls": controls, "selection": selection, "coef_stability": coef_stability,
            "val_result": val_result, "classification": classification, "outdir": outdir, "fold_of": fold_of,
            "cv_gain": cv_gain, "sel_val": sel_val, "g0_val": g0_val, "q0_mean": q0_mean, "shuf_mean": shuf_mean}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Context-free low-capacity PolyU Cross fusion probe (TRAIN CV + one VAL eval).")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--p2_dir", type=str, default=str(Path(DEFAULT_OUTDIR) / "p2_scores"))
    return p


def _rp(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        r = run(manifest_dir=_rp(args.data_dir), outdir=_rp(args.outdir), p2_dir=_rp(args.p2_dir))
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n=== PolyU Cross context-free fusion probe complete ===")
    print(f"Output dir: {r['outdir']}")
    print("\nCV summary (mean AUC per group):")
    print(r["summary"][["group", "mean_auc", "std_auc", "min_auc", "max_auc", "n_folds_beating_G0"]].to_string(index=False))
    print(f"\nControls: Q0 mean={r['q0_mean']:.4f} | shuffled mean={r['shuf_mean']:.4f}")
    print(f"Selected: {r['selection']['selected_group']} (fell_back_to_G0={r['selection']['fell_back_to_G0']})")
    print(f"VAL: selected={r['sel_val']:.4f} G0={r['g0_val']:.4f}")
    print(f"Classification: {r['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
