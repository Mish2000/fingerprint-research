"""Frozen statistical-fusion transfer audit for PolyU Cross (Phase 4A.2B-1).

Read-only audit answering: do the existing SD300-trained statistical fusion
models transfer, frozen, to PolyU CL->CB?

It (1) inventories every SD300 fusion/ablation model with its artifact path,
SHA256, and exact feature schema; (2) determines whether each model's required
feature schema can be constructed *exactly* for PolyU; (3) quantifies the
SD300->PolyU numeric feature-distribution shift and categorical unseen-category
rate; and (4) reports individual-component PolyU CL->CB AUCs.

No model is refit/calibrated and no frozen model is forced to consume
semantically incompatible features. If a model requires unseen PolyU categorical
values (e.g. dataset=polyu_cross, frgp unknown), it is reported as
SCHEMA_INCOMPATIBLE rather than run by silently zeroing one-hot columns.

TRAIN/VAL only. TEST is never read. Manifests, pairs, and model artifacts are
read-only and left byte-identical.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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

from src.fpbench.datasets.polyu_cross_pairs import DATASET as POLYU_CROSS_DATASET
from src.fpbench.universal.quality import extract_image_quality, IMAGE_QUALITY_FEATURES
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_frozen_fusion_transfer_v0"
DEFAULT_SD300_TRAIN_TABLE = (
    "artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_full_pairs/model/training_feature_table.csv"
)
RUN_SCHEMA_VERSION = "polyu_cross_frozen_fusion_transfer_v0"
ALLOWED_SPLITS = ("train", "val")

# PolyU CL->CB per-pair score sources (already produced in earlier phases).
SOURCEAFIS_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0_sourceafis_real"
SIFT_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0"
DEEP_DIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_deep_v0"

# The task's named ablation ladder (primary focus for the inventory).
NAMED_MODELS = [
    "sourceafis_only_calibrated",
    "sourceafis_plus_sift_score",
    "sourceafis_sift_score",
    "sourceafis_plus_sift_geometry",
    "sourceafis_sift_geometry",
    "sourceafis_sift_quality",
    "sourceafis_plus_sift_quality_full",
    "sourceafis_sift_deep_logit",
    "sourceafis_sift_deep_score",
    "sourceafis_sift_quality_deep_fusion_v2",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2",
]


class FusionTransferAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model inventory + schema compatibility
# ---------------------------------------------------------------------------
def read_model_schema(model_dir: Path) -> dict[str, Any]:
    fs = model_dir / "feature_schema.json"
    fm = model_dir / "feature_manifest.json"
    if fs.exists():
        s = json.loads(fs.read_text(encoding="utf-8"))
        return {
            "bundle_format": "v1_feature_schema",
            "numeric_features": s.get("numeric_features", []),
            "categorical_features": s.get("categorical_features", []),
            "categorical_levels": s.get("categorical_levels", {}),
            "schema_file": str(fs),
        }
    if fm.exists():
        s = json.loads(fm.read_text(encoding="utf-8"))
        return {
            "bundle_format": "v2_feature_manifest",
            "numeric_features": s.get("numeric_features", []),
            "categorical_features": s.get("categorical_features", []),
            "categorical_levels": {},  # v2 fits levels from its training table at load
            "schema_file": str(fm),
        }
    return {"bundle_format": "unknown", "numeric_features": [], "categorical_features": [], "categorical_levels": {}, "schema_file": ""}


def assess_compatibility(schema: dict[str, Any], *, constructible_numeric: set[str], polyu_categoricals: dict[str, set[str]], sd300_categorical_levels: dict[str, set[str]]) -> dict[str, Any]:
    """Decide whether the PolyU feature table can satisfy this model's schema exactly."""
    reasons: list[str] = []

    # Numeric: every required numeric feature must be constructible (no zero-fill).
    missing_numeric = [c for c in schema["numeric_features"] if c not in constructible_numeric]
    if missing_numeric:
        reasons.append(f"numeric features not constructible for PolyU (would require zero/median fill): {missing_numeric}")

    # Categorical: PolyU values must be within the model's trained levels.
    unseen: dict[str, Any] = {}
    for cat in schema["categorical_features"]:
        trained = sd300_categorical_levels.get(cat, set())
        polyu_vals = polyu_categoricals.get(cat, set())
        # A model requiring this categorical needs PolyU values inside trained levels.
        new_vals = sorted(v for v in polyu_vals if v not in trained) if trained else sorted(polyu_vals)
        if new_vals:
            unseen[cat] = {"trained": sorted(trained), "polyu": sorted(polyu_vals), "unseen": new_vals}
    if unseen:
        reasons.append(f"categorical features contain unseen PolyU categories: {list(unseen.keys())}")

    compatible = len(reasons) == 0
    return {"compatible": compatible, "verdict": "COMPATIBLE" if compatible else "SCHEMA_INCOMPATIBLE",
            "missing_numeric": missing_numeric, "unseen_categoricals": unseen, "reasons": reasons}


def inventory_models(*, constructible_numeric: set[str], polyu_categoricals: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jb in sorted(glob.glob(str(REPO_ROOT / "artifacts/reports/benchmark/**/model/**/fusion_model.joblib"), recursive=True)):
        model_dir = Path(jb).parent
        schema = read_model_schema(model_dir)
        if schema["bundle_format"] == "unknown":
            continue
        # SD300 trained categorical levels for compatibility (v1 has explicit levels;
        # v2 fits from its training table, which is nist_sd300b/c only).
        sd300_levels = {k: set(v) for k, v in schema["categorical_levels"].items()}
        if schema["bundle_format"] == "v2_feature_manifest":
            sd300_levels = {"dataset": {"nist_sd300b", "nist_sd300c"}, "finger_position": _SD300_FINGER_LEVELS, "frgp": _SD300_FINGER_LEVELS}
        assessment = assess_compatibility(schema, constructible_numeric=constructible_numeric, polyu_categoricals=polyu_categoricals, sd300_categorical_levels=sd300_levels)
        rel = model_dir.relative_to(REPO_ROOT)
        rows.append(
            {
                "model_name": model_dir.name,
                "experiment": model_dir.parent.parent.name,
                "artifact_path": str(rel / "fusion_model.joblib"),
                "sha256": _sha256(model_dir / "fusion_model.joblib"),
                "bundle_format": schema["bundle_format"],
                "n_numeric_features": len(schema["numeric_features"]),
                "categorical_features": ",".join(schema["categorical_features"]),
                "verdict": assessment["verdict"],
                "incompatibility_reasons": " | ".join(assessment["reasons"]),
                "unseen_categoricals": json.dumps(assessment["unseen_categoricals"]),
                "missing_numeric": ",".join(assessment["missing_numeric"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["experiment", "model_name"], kind="mergesort").reset_index(drop=True)


_SD300_FINGER_LEVELS = {"2", "3", "4", "5", "6", "7", "8", "9", "10"}


# ---------------------------------------------------------------------------
# PolyU CL->CB numeric feature table
# ---------------------------------------------------------------------------
def _read_scores(path: Path, cols: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"pair_id": df["pair_id"].astype(str), "label": df["label"].astype(int)})
    for src, dst in cols.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    # keep resolved paths from the sourceafis frame if present
    for p in ("resolved_path_a", "resolved_path_b"):
        if p in df.columns:
            out[p] = df[p].astype(str)
    return out


def build_polyu_feature_table(split: str) -> pd.DataFrame:
    sa = _read_scores(REPO_ROOT / SOURCEAFIS_DIR / f"scores_polyu_cross_sourceafis_open_{split}.csv",
                      {"score": "sourceafis_score"})
    sift = _read_scores(REPO_ROOT / SIFT_DIR / f"scores_polyu_cross_sift_{split}.csv",
                        {"score": "sift_score", "inliers": "sift_inliers", "matches": "sift_matches", "k1": "sift_k1", "k2": "sift_k2"})
    deep = _read_scores(REPO_ROOT / DEEP_DIR / f"scores_polyu_cross_deep_pair_reranker_{split}.csv",
                        {"probability": "deep_score", "logit": "deep_logit"})

    t = sa.merge(sift.drop(columns=[c for c in ("resolved_path_a", "resolved_path_b") if c in sift.columns]), on=["pair_id", "label"], how="inner")
    t = t.merge(deep.drop(columns=[c for c in ("resolved_path_a", "resolved_path_b") if c in deep.columns]), on=["pair_id", "label"], how="inner")
    if not (len(t) == len(sa) == len(sift) == len(deep)):
        raise FusionTransferAuditError(f"Score tables did not align 1:1 for split={split} ({len(sa)},{len(sift)},{len(deep)} -> {len(t)})")

    # Quality features on contactless (a) and contact (b) via the SAME utility as SD300.
    qkeys = [k for k in IMAGE_QUALITY_FEATURES]
    paths = pd.unique(pd.concat([t["resolved_path_a"], t["resolved_path_b"]], ignore_index=True))
    qmap = {p: extract_image_quality(p, repo_root=REPO_ROOT) for p in paths}
    for prefix, col in (("a", "resolved_path_a"), ("b", "resolved_path_b")):
        qdf = pd.DataFrame([{f"{prefix}_{k}": qmap[p][k] for k in qkeys} for p in t[col]])
        t = pd.concat([t.reset_index(drop=True), qdf.reset_index(drop=True)], axis=1)
    for base in ("width", "height", "aspect_ratio", "mean_intensity", "std_intensity", "contrast_proxy", "foreground_ratio", "sharpness_laplacian_var", "edge_density"):
        t[f"pair_{base}_abs_delta"] = (pd.to_numeric(t[f"a_{base}"], errors="coerce") - pd.to_numeric(t[f"b_{base}"], errors="coerce")).abs()
    t["split"] = split
    t["dataset"] = DATASET_NAME
    return t


# ---------------------------------------------------------------------------
# Feature-distribution shift
# ---------------------------------------------------------------------------
def _dist(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("count", "mean", "std", "p01", "p05", "p50", "p95", "p99", "min", "max")}
    q = np.quantile(v, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {"count": int(v.size), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]),
            "min": float(np.min(v)), "max": float(np.max(v))}


def feature_shift_summary(sd300: pd.DataFrame, polyu_tr: pd.DataFrame, polyu_va: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feat in numeric_features:
        sd = _dist(sd300[feat].to_numpy()) if feat in sd300.columns else None
        for split_name, tbl in (("train", polyu_tr), ("val", polyu_va)):
            if feat not in tbl.columns:
                rows.append({"feature": feat, "polyu_split": split_name, "constructible": False, "note": "not constructible for PolyU"})
                continue
            pv = np.asarray(tbl[feat], dtype=float)
            pv_f = pv[np.isfinite(pv)]
            pd_ = _dist(pv)
            row = {"feature": feat, "polyu_split": split_name, "constructible": True}
            for k, val in _dist(sd300[feat].to_numpy() if (sd is not None) else np.array([])).items():
                row[f"sd300_{k}"] = val
            for k, val in pd_.items():
                row[f"polyu_{k}"] = val
            if sd is not None and np.isfinite(sd["std"]) and sd["std"] > 0 and pv_f.size:
                row["standardized_mean_shift"] = (pd_["mean"] - sd["mean"]) / sd["std"]
                row["frac_polyu_below_sd300_p01"] = float(np.mean(pv_f < sd["p01"]))
                row["frac_polyu_above_sd300_p99"] = float(np.mean(pv_f > sd["p99"]))
            else:
                row["standardized_mean_shift"] = float("nan")
                row["frac_polyu_below_sd300_p01"] = float("nan")
                row["frac_polyu_above_sd300_p99"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def categorical_shift(sd300: pd.DataFrame, polyu_tr: pd.DataFrame, polyu_va: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat in categorical_features:
        trained = sorted(set(sd300[cat].astype(str))) if cat in sd300.columns else []
        for split_name, tbl in (("train", polyu_tr), ("val", polyu_va)):
            if cat not in tbl.columns:
                rows.append({"feature": cat, "polyu_split": split_name, "training_categories": trained, "polyu_categories": [], "unseen_count": 0, "unseen_rate": float("nan"), "note": "PolyU has no such column"})
                continue
            polyu_vals = tbl[cat].astype(str)
            unseen_mask = ~polyu_vals.isin(set(trained))
            rows.append({"feature": cat, "polyu_split": split_name, "training_categories": trained,
                         "polyu_categories": sorted(set(polyu_vals)), "unseen_count": int(unseen_mask.sum()),
                         "unseen_rate": float(unseen_mask.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Individual-component metrics
# ---------------------------------------------------------------------------
def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(values)
    labels, values = np.asarray(labels)[finite], np.asarray(values)[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, values))
    except ValueError:
        return float("nan")


def component_metrics(table: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    labels = table["label"].astype(int).to_numpy()
    out = []
    for comp in ("sourceafis_score", "sift_score", "deep_logit", "deep_score"):
        v = pd.to_numeric(table[comp], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(v)
        g = v[finite & (labels == 1)]; im = v[finite & (labels == 0)]
        out.append({
            "component": comp, "split": split, "pair_count": int(len(labels)),
            "scored_count": int(finite.sum()), "failed_count": int((~finite).sum()),
            "auc_raw": _auc(labels, v),
            "genuine_mean": float(np.mean(g)) if g.size else float("nan"),
            "genuine_median": float(np.median(g)) if g.size else float("nan"),
            "genuine_std": float(np.std(g, ddof=1)) if g.size > 1 else float("nan"),
            "impostor_mean": float(np.mean(im)) if im.size else float("nan"),
            "impostor_median": float(np.median(im)) if im.size else float("nan"),
            "impostor_std": float(np.std(im, ddof=1)) if im.size > 1 else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(*, outdir: Path, sd300_train_table: Path) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(sd300_train_table).exists():
        raise FusionTransferAuditError(f"SD300 training feature table not found: {sd300_train_table}")
    sd300 = pd.read_csv(sd300_train_table)

    polyu_tr = build_polyu_feature_table("train")
    polyu_va = build_polyu_feature_table("val")

    constructible_numeric = set(c for c in polyu_tr.columns if polyu_tr[c].dtype.kind in "fi" and c not in ("label",))
    polyu_categoricals = {
        "dataset": set(polyu_tr["dataset"].astype(str)) | set(polyu_va["dataset"].astype(str)),
        "finger_position": {"0"}, "frgp": {"0"},  # PolyU frgp unknown/constant 0
    }
    inv = inventory_models(constructible_numeric=constructible_numeric, polyu_categoricals=polyu_categoricals)
    inv.to_csv(outdir / "model_inventory.csv", index=False)

    # Feature-shift summary over the SD300 v2 numeric feature set (superset).
    v2_numeric = [
        "sourceafis_score", "source_dpi_a", "source_dpi_b", "sift_score", "sift_inliers", "sift_matches", "sift_k1", "sift_k2",
        *[f"a_{k}" for k in IMAGE_QUALITY_FEATURES], *[f"b_{k}" for k in IMAGE_QUALITY_FEATURES],
        *[f"pair_{b}_abs_delta" for b in ("width", "height", "aspect_ratio", "mean_intensity", "std_intensity", "contrast_proxy", "foreground_ratio", "sharpness_laplacian_var", "edge_density")],
        "deep_score", "deep_logit",
    ]
    shift = feature_shift_summary(sd300, polyu_tr, polyu_va, v2_numeric)
    cat_shift = categorical_shift(sd300, polyu_tr, polyu_va, ["dataset", "finger_position", "frgp"])
    shift.to_csv(outdir / "feature_shift_summary.csv", index=False)
    cat_shift.to_csv(outdir / "categorical_shift_summary.csv", index=False)

    # Individual-component metrics (fusion models are schema-incompatible -> not scored).
    comp_rows = component_metrics(polyu_tr, "train") + component_metrics(polyu_va, "val")
    metrics = pd.DataFrame(comp_rows)
    fusion_incompatible = inv[inv["verdict"] == "SCHEMA_INCOMPATIBLE"]
    metrics_csv = outdir / "fusion_transfer_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    # Compact VAL comparison: individual components + fusion (schema-incompatible).
    val = metrics[metrics["split"] == "val"][["component", "auc_raw"]].rename(columns={"component": "system", "auc_raw": "val_auc"})
    val["kind"] = "individual_component"
    fusion_named = inv[inv["model_name"].isin(NAMED_MODELS)][["model_name", "verdict"]].drop_duplicates("model_name")
    fusion_rows = pd.DataFrame({"system": fusion_named["model_name"], "val_auc": np.nan, "kind": "frozen_fusion_" + fusion_named["verdict"].str.lower()})
    val_cmp = pd.concat([val, fusion_rows], ignore_index=True)
    val_cmp.to_csv(outdir / "val_comparison.csv", index=False)

    # Classification.
    n_compatible = int((inv["verdict"] == "COMPATIBLE").sum())
    classification = "D. SCHEMA_INCOMPATIBLE" if n_compatible == 0 else "SEE_METRICS"

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "protocol": "TRAIN/VAL only; TEST never read. Read-only frozen transfer audit.",
        "sd300_training_feature_table": {"path": str(sd300_train_table), "sha256": _sha256(Path(sd300_train_table)), "rows": int(len(sd300)),
                                          "datasets": sorted(set(sd300["dataset"].astype(str))) if "dataset" in sd300 else []},
        "polyu_feature_table": {"train_rows": int(len(polyu_tr)), "val_rows": int(len(polyu_va)),
                                 "constructible_numeric": sorted(constructible_numeric),
                                 "unavailable_numeric": ["source_dpi_a", "source_dpi_b"],
                                 "note": "PolyU sourceafis scores carry no per-image DPI (dpi_strategy=default); source_dpi_* not constructible."},
        "n_models_inventoried": int(len(inv)),
        "n_models_compatible": n_compatible,
        "n_models_schema_incompatible": int((inv["verdict"] == "SCHEMA_INCOMPATIBLE").sum()),
        "classification": classification,
        "incompatibility_summary": (
            "Every SD300 fusion model uses categorical features dataset/finger_position/frgp. PolyU dataset=polyu_cross "
            "is unseen (trained on nist_sd300b/c) and PolyU frgp is unknown/constant 0 (trained finger positions 2..10). "
            "There is no numeric-only fusion model. Running any model would require silently zeroing unseen one-hot "
            "columns (hacking the encoder), which is disallowed; the models are therefore schema-incompatible for "
            "faithful RAW frozen transfer."
        ),
        "reused_polyu_scores": {"sourceafis": SOURCEAFIS_DIR, "sift": SIFT_DIR, "deep": DEEP_DIR,
                                 "note": "PolyU 'sift' is the basic sift method; SD300 fusion 'sift_score' provenance may differ in config (additional caveat)."},
        "outputs": {"model_inventory_csv": str(outdir / "model_inventory.csv"),
                     "feature_shift_summary_csv": str(outdir / "feature_shift_summary.csv"),
                     "categorical_shift_summary_csv": str(outdir / "categorical_shift_summary.csv"),
                     "fusion_transfer_metrics_csv": str(metrics_csv),
                     "val_comparison_csv": str(outdir / "val_comparison.csv")},
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "constraints": {"trained_or_refit_fusion": False, "calibrated_on_polyu": False, "selected_thresholds": False,
                        "evaluated_test": False, "modified_manifest_or_pairs": False, "modified_fusion_artifacts": False,
                        "ran_frozen_model_on_incompatible_features": False, "silently_zero_filled_features": False},
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {"inventory": inv, "metrics": metrics, "val": val_cmp, "shift": shift, "cat_shift": cat_shift,
            "classification": classification, "manifest_json": manifest_json, "outdir": outdir,
            "polyu_tr": polyu_tr, "polyu_va": polyu_va, "sd300": sd300}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen fusion transfer audit for PolyU Cross (read-only, TRAIN/VAL).")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--sd300_train_table", type=str, default=DEFAULT_SD300_TRAIN_TABLE)
    return p


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(outdir=_resolve_repo_path(args.outdir), sd300_train_table=_resolve_repo_path(args.sd300_train_table))
    except FusionTransferAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    inv = result["inventory"]
    print("\n=== PolyU Cross frozen fusion transfer audit complete ===")
    print(f"Output dir: {result['outdir']}")
    print(f"Models inventoried: {len(inv)} | compatible: {int((inv['verdict']=='COMPATIBLE').sum())} | schema-incompatible: {int((inv['verdict']=='SCHEMA_INCOMPATIBLE').sum())}")
    print(f"Classification: {result['classification']}")
    print("\nIndividual-component VAL AUC (CL->CB):")
    print(result["metrics"][result["metrics"].split == "val"][["component", "auc_raw"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
