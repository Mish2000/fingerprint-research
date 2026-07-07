"""PolyU Cross modality-control decomposition (Phase 4A.1B) - diagnostics only.

Builds deterministic *within-modality* control verification protocols from the
existing hardened PolyU Cross TRAIN/VAL finger-unit assignments (read from the
manifest ``split`` column - never regenerated) and scores them with the exact
existing frozen scorers:

* real ``sourceafis_open`` (reuses ``run_polyu_cross_zero_shot.score_split_sourceafis``)
* the frozen SD300 deep pair reranker (reuses ``run_polyu_cross_deep_reranker``)

Protocols built here (TRAIN/VAL only):
  1. contact_based_to_contact_based_same_session
  2. contact_based_to_contact_based_cross_session
  3. contactless_to_contactless_same_session
  4. contactless_to_contactless_cross_session
The existing cross-modality ``contactless_to_contact_based`` protocol (5) is
*reused* from the Phase 3A.1 / 3A.2 score outputs and never regenerated.

Strictly diagnostic. No training, fine-tuning, calibration, thresholding,
TEST usage, manifest/pair mutation, preprocessing changes, or UI/API work.
Identity unit is ``finger_unit_id`` (hardened PolyU semantics); FRGP/anatomical
finger positions are not invented (frgp stays 0/unknown).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    resolve_pairs_frame,
    resolve_polyu_cross_root,
)
from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, sha256_file, utc_now

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
RUN_SCHEMA_VERSION = "polyu_cross_modality_controls_v0"

ALLOWED_SPLITS = ("train", "val")
CONTACT = "contact_based_2d"
CONTACTLESS = "contactless_2d"
DEFAULT_BASE_SEED = 42
DEFAULT_NEG_PER_POS = 3
DEFAULT_MAX_POS = 300

# (protocol_id, modality, session_relation)
CONTROL_PROTOCOLS = [
    ("contact_based_to_contact_based_same_session", CONTACT, "same"),
    ("contact_based_to_contact_based_cross_session", CONTACT, "cross"),
    ("contactless_to_contactless_same_session", CONTACTLESS, "same"),
    ("contactless_to_contactless_cross_session", CONTACTLESS, "cross"),
]
EXISTING_CROSS_MODALITY_PROTOCOL = "contactless_to_contact_based"

PAIR_COLUMNS = [
    "protocol_id", "split", "pair_id", "label",
    "subject_a", "subject_b", "finger_unit_a", "finger_unit_b", "frgp",
    "sample_uid_a", "sample_uid_b", "modality_a", "modality_b",
    "session_a", "session_b", "path_a", "path_b",
]


class ModalityControlError(RuntimeError):
    """Raised for unrecoverable control-protocol setup/protocol failures."""


# ---------------------------------------------------------------------------
# Manifest image pool (TRAIN/VAL only)
# ---------------------------------------------------------------------------
def load_manifest_images(manifest_dir: Path) -> pd.DataFrame:
    manifest_csv = Path(manifest_dir) / "manifest.csv"
    if not manifest_csv.exists():
        raise ModalityControlError(f"PolyU Cross manifest.csv not found: {manifest_csv}")
    m = pd.read_csv(manifest_csv, dtype=str)
    required = {"finger_unit_id", "sample_uid", "modality", "session_id", "split", "path"}
    missing = sorted(required - set(m.columns))
    if missing:
        raise ModalityControlError(f"manifest.csv missing columns {missing}; found {list(m.columns)}")
    m = m[m["split"].isin(ALLOWED_SPLITS)].copy()  # TRAIN/VAL only; TEST never read
    m = m[["finger_unit_id", "sample_uid", "modality", "session_id", "split", "path"]]
    m = m.sort_values(["split", "modality", "finger_unit_id", "session_id", "sample_uid"], kind="mergesort")
    return m.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deterministic pair construction
# ---------------------------------------------------------------------------
def _canonical_pair(u1: dict, u2: dict) -> tuple[dict, dict]:
    """Order two image records deterministically by sample_uid (unordered form)."""
    return (u1, u2) if str(u1["sample_uid"]) <= str(u2["sample_uid"]) else (u2, u1)


def _positive_candidates_for_fu(records: list[dict], relation: str) -> list[tuple[dict, dict]]:
    """All valid positive image pairs for one finger_unit under a session relation."""
    pairs: list[tuple[dict, dict]] = []
    recs = sorted(records, key=lambda r: str(r["sample_uid"]))
    n = len(recs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = recs[i], recs[j]
            same_session = str(a["session_id"]) == str(b["session_id"])
            if relation == "same" and not same_session:
                continue
            if relation == "cross" and same_session:
                continue
            pairs.append(_canonical_pair(a, b))
    return pairs


def build_control_pairs(
    images: pd.DataFrame,
    *,
    protocol_id: str,
    modality: str,
    relation: str,
    split: str,
    max_pos: int,
    neg_per_pos: int,
    base_seed: int,
) -> pd.DataFrame:
    pool = images[(images["split"] == split) & (images["modality"] == modality)]
    records = pool.to_dict("records")
    by_fu: dict[str, list[dict]] = {}
    for rec in records:
        by_fu.setdefault(str(rec["finger_unit_id"]), []).append(rec)

    # Positives: round-robin across finger_units for identity balance (deterministic).
    fu_candidates: dict[str, list[tuple[dict, dict]]] = {
        fu: _positive_candidates_for_fu(recs, relation) for fu, recs in by_fu.items()
    }
    fu_order = sorted(fu for fu, cands in fu_candidates.items() if cands)
    used: set[frozenset] = set()
    positives: list[tuple[dict, dict]] = []
    cursor = {fu: 0 for fu in fu_order}
    exhausted = False
    while len(positives) < int(max_pos) and not exhausted:
        exhausted = True
        for fu in fu_order:
            if len(positives) >= int(max_pos):
                break
            cands = fu_candidates[fu]
            idx = cursor[fu]
            while idx < len(cands):
                a, b = cands[idx]
                idx += 1
                key = frozenset({str(a["sample_uid"]), str(b["sample_uid"])})
                if key in used:
                    continue
                used.add(key)
                positives.append((a, b))
                exhausted = False
                break
            cursor[fu] = idx

    # Negatives: per positive anchor (image a), sample neg_per_pos partners from a
    # different finger_unit matching the protocol's session relation. Deterministic
    # RNG; no self-pairs, no duplicate unordered pairs.
    seed = (int(base_seed) * 1_000_003 + abs(hash((protocol_id, split))) % 1_000_003) % (2**32)
    rng = np.random.default_rng(seed)
    pool_records = records
    pool_uid = np.array([str(r["sample_uid"]) for r in pool_records])
    pool_fu = np.array([str(r["finger_unit_id"]) for r in pool_records])
    pool_ses = np.array([str(r["session_id"]) for r in pool_records])

    negatives: list[tuple[dict, dict]] = []
    for a, _b in positives:
        a_fu = str(a["finger_unit_id"])
        a_ses = str(a["session_id"])
        if relation == "same":
            mask = (pool_fu != a_fu) & (pool_ses == a_ses)
        else:
            mask = (pool_fu != a_fu) & (pool_ses != a_ses)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            continue
        order = rng.permutation(cand_idx.size)
        picked = 0
        for oi in order:
            partner = pool_records[int(cand_idx[int(oi)])]
            key = frozenset({str(a["sample_uid"]), str(partner["sample_uid"])})
            if str(partner["sample_uid"]) == str(a["sample_uid"]) or key in used:
                continue
            used.add(key)
            negatives.append(_canonical_pair(a, partner))
            picked += 1
            if picked >= int(neg_per_pos):
                break

    rows: list[dict[str, Any]] = []

    def _emit(pair: tuple[dict, dict], label: int) -> None:
        a, b = pair
        rows.append(
            {
                "protocol_id": protocol_id,
                "split": split,
                "label": int(label),
                "subject_a": str(a["finger_unit_id"]),
                "subject_b": str(b["finger_unit_id"]),
                "finger_unit_a": str(a["finger_unit_id"]),
                "finger_unit_b": str(b["finger_unit_id"]),
                "frgp": 0,
                "sample_uid_a": str(a["sample_uid"]),
                "sample_uid_b": str(b["sample_uid"]),
                "modality_a": str(a["modality"]),
                "modality_b": str(b["modality"]),
                "session_a": str(a["session_id"]),
                "session_b": str(b["session_id"]),
                "path_a": str(a["path"]),
                "path_b": str(b["path"]),
            }
        )

    for pair in positives:
        _emit(pair, 1)
    for pair in negatives:
        _emit(pair, 0)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    # Deterministic order: positives first, then negatives; stable within.
    df = df.sort_values(["label", "sample_uid_a", "sample_uid_b"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    df["pair_id"] = [f"{protocol_id}|{split}|{i:06d}" for i in range(len(df))]
    return df[PAIR_COLUMNS]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check_pairs(df: pd.DataFrame, *, modality: str, relation: str, split: str, test_uids: set[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if df.empty:
        raise ModalityControlError(f"No pairs generated for protocol modality={modality} relation={relation} split={split}")

    report["no_test_samples"] = not (
        set(df["sample_uid_a"]).intersection(test_uids) or set(df["sample_uid_b"]).intersection(test_uids)
    )
    report["single_split"] = set(df["split"].unique()) == {split}
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    report["positives_same_finger_unit"] = bool((pos["finger_unit_a"].astype(str) == pos["finger_unit_b"].astype(str)).all())
    report["negatives_diff_finger_unit"] = bool((neg["finger_unit_a"].astype(str) != neg["finger_unit_b"].astype(str)).all())
    report["no_self_pairs"] = bool((df["sample_uid_a"].astype(str) != df["sample_uid_b"].astype(str)).all())
    unordered = df.apply(lambda r: frozenset({str(r["sample_uid_a"]), str(r["sample_uid_b"])}), axis=1)
    report["no_duplicate_unordered_pairs"] = bool(unordered.nunique() == len(df))
    report["modality_constraint"] = bool(((df["modality_a"] == modality) & (df["modality_b"] == modality)).all())
    if relation == "same":
        report["session_constraint"] = bool((df["session_a"].astype(str) == df["session_b"].astype(str)).all())
    else:
        report["session_constraint"] = bool((df["session_a"].astype(str) != df["session_b"].astype(str)).all())
    report["n_pairs"] = int(len(df))
    report["n_positive"] = int(len(pos))
    report["n_negative"] = int(len(neg))

    failed = {k: v for k, v in report.items() if isinstance(v, bool) and not v}
    if failed:
        raise ModalityControlError(f"Sanity check(s) failed for {modality}/{relation}/{split}: {sorted(failed)}")
    return report


# ---------------------------------------------------------------------------
# Metrics (reuse existing AUC/EER implementation)
# ---------------------------------------------------------------------------
def score_metrics(labels: np.ndarray, scores: np.ndarray, *, want_eer: bool) -> dict[str, Any]:
    from pipelines.benchmark.evaluate import compute_auc_eer

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    n_scored = int(finite.sum())
    n_failed = int((~finite).sum())
    gen = scores[finite & (labels == 1)]
    imp = scores[finite & (labels == 0)]

    def _stats(a: np.ndarray) -> dict[str, float]:
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0, "median": float(np.median(a))}

    metrics = compute_auc_eer(labels[finite], scores[finite]) if n_scored else None
    out = {
        "n_pairs": int(len(labels)),
        "genuine_count": int((labels == 1).sum()),
        "impostor_count": int((labels == 0).sum()),
        "scored_count": n_scored,
        "failed_count": n_failed,
        "genuine_mean": _stats(gen)["mean"],
        "genuine_std": _stats(gen)["std"],
        "genuine_median": _stats(gen)["median"],
        "impostor_mean": _stats(imp)["mean"],
        "impostor_std": _stats(imp)["std"],
        "impostor_median": _stats(imp)["median"],
        "roc_auc": float(metrics.auc) if metrics is not None else float("nan"),
    }
    out["eer"] = float(metrics.eer) if (want_eer and metrics is not None) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Scoring (reuse existing frozen implementations)
# ---------------------------------------------------------------------------
def _resolve(df: pd.DataFrame, root: Optional[Path]) -> pd.DataFrame:
    resolved = resolve_pairs_frame(df.copy(), root)
    missing = resolved[(~resolved["path_a_exists"]) | (~resolved["path_b_exists"])]
    if not missing.empty:
        raise ModalityControlError(f"{len(missing)} control pair(s) reference missing images; first: {missing.iloc[0]['resolved_path_a']!r}")
    return resolved


def score_sourceafis(resolved: pd.DataFrame, *, split: str, work_dir: Path) -> pd.DataFrame:
    from pipelines.benchmark.run_polyu_cross_zero_shot import score_split_sourceafis

    enriched, _timing = score_split_sourceafis(split=split, resolved_df=resolved, work_dir=work_dir)
    return enriched


def score_deep(frozen: Any, resolved: pd.DataFrame) -> np.ndarray:
    from pipelines.benchmark.run_polyu_cross_deep_reranker import score_resolved_pairs

    _logits, probs, _timing = score_resolved_pairs(frozen, resolved.reset_index(drop=True))
    return np.asarray(probs, dtype=float)


# ---------------------------------------------------------------------------
# Existing CL->CB reuse
# ---------------------------------------------------------------------------
def load_existing_scores(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "label" not in df.columns or "score" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class ProtocolResult:
    protocol_id: str
    split: str
    pairs_csv: Optional[Path]
    n_pairs: int
    n_positive: int
    n_negative: int


def run(
    *,
    manifest_dir: Path,
    outdir: Path,
    checkpoint: Path,
    max_pos: int,
    neg_per_pos: int,
    base_seed: int,
    polyu_root: Optional[str],
    sourceafis_clcb_dir: Path,
    deep_clcb_dir: Path,
    score_methods: tuple[str, ...] = ("sourceafis_open", "deep_pair_reranker_fast_ddp"),
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs_dir = outdir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    work_dir = outdir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    images = load_manifest_images(manifest_dir)
    # TEST uids collected ONLY to assert they never appear (not used for pairs).
    test_uids = set(
        pd.read_csv(Path(manifest_dir) / "manifest.csv", dtype=str).query("split == 'test'")["sample_uid"].astype(str)
    )
    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)

    # --- Build + sanity-check control pairs ---
    control_pairs: dict[tuple[str, str], pd.DataFrame] = {}
    sanity: dict[str, Any] = {}
    for protocol_id, modality, relation in CONTROL_PROTOCOLS:
        for split in ALLOWED_SPLITS:
            df = build_control_pairs(
                images,
                protocol_id=protocol_id,
                modality=modality,
                relation=relation,
                split=split,
                max_pos=max_pos,
                neg_per_pos=neg_per_pos,
                base_seed=base_seed,
            )
            report = sanity_check_pairs(df, modality=modality, relation=relation, split=split, test_uids=test_uids)
            sanity[f"{protocol_id}|{split}"] = report
            pairs_csv = pairs_dir / f"pairs_{protocol_id}_{split}.csv"
            df.to_csv(pairs_csv, index=False)
            control_pairs[(protocol_id, split)] = df

    # --- Frozen deep model (loaded once) ---
    frozen = None
    deep_provenance: dict[str, Any] = {}
    if "deep_pair_reranker_fast_ddp" in score_methods:
        from pipelines.benchmark.run_polyu_cross_deep_reranker import load_frozen_model

        frozen = load_frozen_model(Path(checkpoint), device_arg="auto")
        deep_provenance = frozen.provenance

    # --- Score control protocols ---
    metric_rows: list[dict[str, Any]] = []
    for (protocol_id, split), df in control_pairs.items():
        resolved = _resolve(df, resolved_root.root)
        labels = df["label"].astype(int).to_numpy()
        if "sourceafis_open" in score_methods:
            enriched = score_sourceafis(resolved, split=split, work_dir=work_dir)
            scores = pd.to_numeric(enriched["score"], errors="coerce").to_numpy(dtype=float)
            metric_rows.append(
                {"method": "sourceafis_open", "protocol": protocol_id, "split": split, **score_metrics(labels, scores, want_eer=(split == "val"))}
            )
        if frozen is not None:
            probs = score_deep(frozen, resolved)
            metric_rows.append(
                {"method": "deep_pair_reranker_fast_ddp", "protocol": protocol_id, "split": split, **score_metrics(labels, probs, want_eer=(split == "val"))}
            )

    # --- Reuse existing CL->CB scores (protocol 5), never regenerated ---
    clcb_sources = {
        "sourceafis_open": sourceafis_clcb_dir / "scores_polyu_cross_sourceafis_open_{split}.csv",
        "deep_pair_reranker_fast_ddp": deep_clcb_dir / "scores_polyu_cross_deep_pair_reranker_{split}.csv",
    }
    clcb_reused: dict[str, Any] = {}
    for method, template in clcb_sources.items():
        if method not in score_methods and not (method == "deep_pair_reranker_fast_ddp" and frozen is not None):
            continue
        for split in ALLOWED_SPLITS:
            path = Path(str(template).format(split=split))
            existing = load_existing_scores(path)
            clcb_reused[f"{method}|{split}"] = str(path) if existing is not None else None
            if existing is None:
                continue
            labels = existing["label"].astype(int).to_numpy()
            scores = pd.to_numeric(existing["score"], errors="coerce").to_numpy(dtype=float)
            metric_rows.append(
                {"method": method, "protocol": EXISTING_CROSS_MODALITY_PROTOCOL, "split": split, **score_metrics(labels, scores, want_eer=(split == "val"))}
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["method", "protocol", "split"], kind="mergesort").reset_index(drop=True)
    metrics_csv = outdir / "control_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Compact method,protocol,val_auc table.
    protocol_order = [p[0] for p in CONTROL_PROTOCOLS] + [EXISTING_CROSS_MODALITY_PROTOCOL]
    val = metrics_df[metrics_df["split"] == "val"][["method", "protocol", "roc_auc"]].copy()
    val = val.rename(columns={"roc_auc": "val_auc"})
    val["_p"] = val["protocol"].apply(lambda x: protocol_order.index(x) if x in protocol_order else 99)
    val = val.sort_values(["method", "_p"], kind="mergesort").drop(columns="_p").reset_index(drop=True)
    val_auc_csv = outdir / "val_auc_matrix.csv"
    val.to_csv(val_auc_csv, index=False)

    pair_counts = [
        {
            "protocol": protocol_id,
            "split": split,
            "n_pairs": int(len(df)),
            "n_positive": int((df["label"] == 1).sum()),
            "n_negative": int((df["label"] == 0).sum()),
        }
        for (protocol_id, split), df in control_pairs.items()
    ]

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "manifest_dir": str(manifest_dir),
        "outdir": str(outdir),
        "splits_evaluated": list(ALLOWED_SPLITS),
        "protocol": "TRAIN and VAL only; TEST never read for pair construction or scoring.",
        "identity_unit": "finger_unit_id",
        "control_protocols": [p[0] for p in CONTROL_PROTOCOLS],
        "reused_cross_modality_protocol": EXISTING_CROSS_MODALITY_PROTOCOL,
        "reused_clcb_score_sources": clcb_reused,
        "pair_policy": {
            "max_pos_per_protocol_split": int(max_pos),
            "neg_per_pos": int(neg_per_pos),
            "base_seed": int(base_seed),
            "positive_rule": "same finger_unit_id, distinct sample_uid, protocol modality+session relation",
            "negative_rule": "different finger_unit_id, same split+modality, same session relation, deterministic without replacement, no duplicate unordered pairs",
        },
        "pair_counts": pair_counts,
        "sanity_checks": sanity,
        "polyu_root": {"path": str(resolved_root.root) if resolved_root.root else None, "source": resolved_root.source, "exists": bool(resolved_root.exists)},
        "deep_model_provenance": deep_provenance,
        "outputs": {"control_metrics_csv": str(metrics_csv), "val_auc_matrix_csv": str(val_auc_csv)},
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "constraints": {
            "trained_or_finetuned": False,
            "calibrated_scores": False,
            "selected_thresholds": False,
            "evaluated_test": False,
            "modified_manifest_or_pairs": False,
            "regenerated_canonical_clcb_pairs": False,
            "instantiated_optimizer_or_scheduler": False,
            "altered_sourceafis_params": False,
            "altered_deep_preprocessing": False,
            "added_preprocessing_transformations": False,
            "modified_ui_or_api": False,
            "copied_biometric_images": False,
        },
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {"metrics": metrics_df, "val_auc": val, "pair_counts": pair_counts, "sanity": sanity, "manifest_json": manifest_json, "outdir": outdir}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PolyU Cross modality-control decomposition (diagnostics, TRAIN/VAL only).")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--max_pos", type=int, default=DEFAULT_MAX_POS, help="Max positive pairs per protocol per split (balanced across finger_units).")
    p.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    p.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--methods", type=str, default="sourceafis_open,deep_pair_reranker_fast_ddp")
    p.add_argument("--sourceafis_clcb_dir", type=str, default="artifacts/reports/benchmark/polyu_cross_zero_shot_v0_sourceafis_real")
    p.add_argument("--deep_clcb_dir", type=str, default="artifacts/reports/benchmark/polyu_cross_zero_shot_deep_v0")
    return p


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = tuple(s.strip() for s in str(args.methods).split(",") if s.strip())
    try:
        result = run(
            manifest_dir=_resolve_repo_path(args.data_dir),
            outdir=_resolve_repo_path(args.outdir),
            checkpoint=_resolve_repo_path(args.checkpoint),
            max_pos=int(args.max_pos),
            neg_per_pos=int(args.neg_per_pos),
            base_seed=int(args.base_seed),
            polyu_root=str(args.polyu_root).strip() or None,
            sourceafis_clcb_dir=_resolve_repo_path(args.sourceafis_clcb_dir),
            deep_clcb_dir=_resolve_repo_path(args.deep_clcb_dir),
            score_methods=methods,
        )
    except ModalityControlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n=== PolyU Cross modality-control decomposition complete ===")
    print(f"Output dir : {result['outdir']}")
    print("\nControl pair counts:")
    print(pd.DataFrame(result["pair_counts"]).to_string(index=False))
    print("\nmethod,protocol,val_auc:")
    print(result["val_auc"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
