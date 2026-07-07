"""Phase 4B.2A.1 bounded Layer x P2 local-signal probe for PolyU Cross.

This diagnostic is deliberately small and frozen-feature only:
* exactly three topology-selected ConvEncoder stages (E/M/F),
* exactly two contactless-side preprocessing conditions (RAW/P2),
* exactly one matcher: Phase 4B.2A L2 mutual-nearest-neighbor local score,
* TRAIN inner-dev identities only,
* official VAL and TEST never read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, safe_pkg_version, sha256_file, utc_now
from scripts.deep.score_fast_pair_ddp_splits import load_image_u8
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import PolyUCrossPairError


RUN_SCHEMA_VERSION = "polyu_cross_local_layer_p2_probe_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_local_layer_p2_probe_v0"

RAW = "RAW"
P2 = "P2"
PREPROCESS_CONDITIONS = (RAW, P2)
P2_LO_PCT = 5.0
P2_HI_PCT = 95.0
GRID_SIZE = 24
BASELINE_F_RAW_L2_AUC = 0.5119
BASELINE_RETRIEVAL = {
    "CL_probe_to_CB_gallery": {"recall_at_1": 0.050, "recall_at_5": 0.150, "mrr": 0.138137},
    "CB_probe_to_CL_gallery": {"recall_at_1": 0.025, "recall_at_5": 0.125, "mrr": 0.110220},
}


class LayerP2ProbeError(RuntimeError):
    """Raised for protocol or artifact failures in Phase 4B.2A.1."""


@dataclass(frozen=True)
class StageSpec:
    label: str
    stage_index: int
    stage_name: str
    topology_role: str
    selection_rationale: str


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        label="E",
        stage_index=0,
        stage_name="encoder.net.0",
        topology_role="early",
        selection_rationale="First convolutional block output; highest native spatial resolution and lowest RF.",
    ),
    StageSpec(
        label="M",
        stage_index=2,
        stage_name="encoder.net.2",
        topology_role="middle",
        selection_rationale="Middle/deep convolutional block before the final block; balances ridge detail and abstraction.",
    ),
    StageSpec(
        label="F",
        stage_index=3,
        stage_name="encoder.net.3",
        topology_role="final_pre_pool",
        selection_rationale="Phase 4B.2A selected final pre-pooling feature map.",
    ),
)


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 1341
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 64
    score_batch_size: int = 32
    amp: bool = True
    auc_present_gain: float = 0.05
    retrieval_mrr_gain: float = 0.03
    diagnostic_cosine_gap: float = 0.02
    diagnostic_match_count_gap: float = 0.5
    p2_min_stage_improvements: int = 3
    close_to_baseline_auc: float = 0.03


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def p2_robust_intensity_norm(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [P2_LO_PCT, P2_HI_PCT])
    rng = float(hi) - float(lo)
    if rng < 1.0:
        return img.astype(np.uint8)
    out = (img.astype(np.float64) - float(lo)) * (255.0 / rng)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def load_image_u8_for_condition(row: dict[str, Any], *, input_size: int, preprocess: str) -> torch.Tensor:
    path = Path(str(row["resolved_path"]))
    if preprocess == P2 and str(row["modality"]) == base.CONTACTLESS:
        with Image.open(path) as im:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.uint8).copy()
        arr = p2_robust_intensity_norm(arr)
        out = Image.fromarray(arr, mode="L").resize((int(input_size), int(input_size)), Image.BILINEAR)
        return torch.from_numpy(np.asarray(out, dtype=np.uint8).copy()).unsqueeze(0).contiguous()
    return load_image_u8(path, int(input_size))


def load_image_store_for_condition(
    images: pd.DataFrame,
    *,
    input_size: int,
    preprocess: str,
) -> dict[str, torch.Tensor]:
    unique = images.drop_duplicates("sample_uid").sort_values("sample_uid", kind="mergesort")
    store: dict[str, torch.Tensor] = {}
    start = time.perf_counter()
    for i, row in enumerate(unique.to_dict("records")):
        if i and i % 500 == 0:
            print(f"[images:{preprocess}] loaded {i}/{len(unique)} elapsed={time.perf_counter() - start:.1f}s", flush=True)
        store[str(row["sample_uid"])] = load_image_u8_for_condition(row, input_size=input_size, preprocess=preprocess)
    print(f"[images:{preprocess}] loaded {len(store)} TRAIN inner-dev images", flush=True)
    return store


def feature_stage_inventory(*, width: int, input_size: int) -> pd.DataFrame:
    topo = {
        int(row["stage_name"].split(".")[-1]): row
        for row in local.convencoder_feature_map_inventory(width=width, input_size=input_size)
        if str(row["stage_name"]).startswith("encoder.net.") and str(row["module_type"]) == "conv_block_with_final_maxpool"
    }
    rows: list[dict[str, Any]] = []
    for spec in STAGE_SPECS:
        row = topo[spec.stage_index]
        native_h = int(row["spatial_height"])
        native_w = int(row["spatial_width"])
        if native_h > GRID_SIZE or native_w > GRID_SIZE:
            op = f"adaptive_avg_pool2d(({GRID_SIZE},{GRID_SIZE}))"
            out_h = out_w = GRID_SIZE
        elif native_h == GRID_SIZE and native_w == GRID_SIZE:
            op = "preserve_native_24x24"
            out_h = native_h
            out_w = native_w
        else:
            op = "not_allowed_no_upsampling"
            out_h = native_h
            out_w = native_w
        rows.append(
            {
                "stage_label": spec.label,
                "stage_name": spec.stage_name,
                "topology_role": spec.topology_role,
                "selection_rationale": spec.selection_rationale,
                "channels": int(row["channel_count"]),
                "native_spatial_height": native_h,
                "native_spatial_width": native_w,
                "normalized_spatial_height": out_h,
                "normalized_spatial_width": out_w,
                "descriptor_count": int(out_h * out_w),
                "descriptor_dim": int(row["channel_count"]),
                "approx_stride": int(row["effective_stride_pixels"]),
                "approx_receptive_field": int(row["receptive_field_pixels"]),
                "descriptor_grid_operation": op,
                "selected_before_metrics": True,
            }
        )
    return pd.DataFrame(rows)


def normalize_feature_grid(feature_map: torch.Tensor, *, target_size: int = GRID_SIZE) -> tuple[torch.Tensor, str]:
    if feature_map.ndim != 4:
        raise LayerP2ProbeError(f"Expected BCHW feature map; got {tuple(feature_map.shape)}")
    h, w = int(feature_map.shape[2]), int(feature_map.shape[3])
    if h > target_size or w > target_size:
        pooled = F.adaptive_avg_pool2d(feature_map, (target_size, target_size))
        op = f"adaptive_avg_pool2d(({target_size},{target_size}))"
    elif h == target_size and w == target_size:
        pooled = feature_map
        op = "preserve_native_24x24"
    else:
        raise LayerP2ProbeError(f"Refusing to upsample {h}x{w} feature map to {target_size}x{target_size}")
    local_grid = pooled.float().permute(0, 2, 3, 1).contiguous()
    return F.normalize(local_grid, p=2, dim=3), op


@torch.inference_mode()
def extract_stage_descriptor_cache(
    *,
    encoder: nn.Module,
    image_store: dict[str, torch.Tensor],
    uids: list[str],
    device: torch.device,
    batch_size: int,
    stage: StageSpec,
    amp: bool,
) -> tuple[local.DescriptorCache, dict[str, Any]]:
    encoder.eval()
    target = local._target_stage(encoder, stage.stage_index)
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = target.register_forward_hook(_hook)
    local_grids: dict[str, np.ndarray] = {}
    global_embeddings: dict[str, np.ndarray] = {}
    native_h = native_w = channels = 0
    op = ""
    try:
        for start in range(0, len(uids), int(batch_size)):
            chunk = uids[start : start + int(batch_size)]
            captured.clear()
            x = base.stack_batch(image_store, chunk, device=device)
            with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
                emb = encoder(x)
            if len(captured) != 1:
                raise LayerP2ProbeError(f"Expected one captured map for {stage.stage_name}; got {len(captured)}")
            fmap = captured[0].float()
            if native_h == 0:
                native_h, native_w, channels = int(fmap.shape[2]), int(fmap.shape[3]), int(fmap.shape[1])
            local_grid, op = normalize_feature_grid(fmap, target_size=GRID_SIZE)
            global_z = F.normalize(emb.float(), p=2, dim=1)
            grid_np = local_grid.detach().cpu().numpy().astype(np.float32, copy=True)
            global_np = global_z.detach().cpu().numpy().astype(np.float32, copy=True)
            for uid, grid, vec in zip(chunk, grid_np, global_np):
                local_grids[str(uid)] = grid
                global_embeddings[str(uid)] = vec
    finally:
        handle.remove()
    ordered = [str(uid) for uid in uids]
    cache = local.DescriptorCache(
        sample_uids=ordered,
        uid_to_index={uid: i for i, uid in enumerate(ordered)},
        local_grids=local_grids,
        global_embeddings=global_embeddings,
        height=GRID_SIZE,
        width=GRID_SIZE,
        channels=channels,
        selected_stage_name=stage.stage_name,
    )
    meta = {
        "native_height": int(native_h),
        "native_width": int(native_w),
        "normalized_height": GRID_SIZE,
        "normalized_width": GRID_SIZE,
        "channels": int(channels),
        "descriptor_count_per_image": GRID_SIZE * GRID_SIZE,
        "descriptor_grid_operation": op,
    }
    return cache, meta


def _condition_id(stage: StageSpec, preprocess: str) -> str:
    return f"{stage.label}x{preprocess}"


def _score_clcb_condition(
    *,
    stage: StageSpec,
    preprocess: str,
    pair_df: pd.DataFrame,
    retrieval_table: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    score_batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = pair_df.reset_index(drop=True).copy()
    df["protocol_id"] = "contactless_to_contact_based"
    scores, details, elapsed = local.score_pair_frame(
        method=local.L2_MNN,
        df=df,
        tensors=tensors,
        device=device,
        batch_size=score_batch_size,
        l3_shift_radius=0,
    )
    metric = local.metric_row_from_scores(
        method=local.L2_MNN,
        stage="inner_dev",
        protocol="contactless_to_contact_based",
        df=df,
        scores=scores,
        elapsed_seconds=elapsed,
    )
    metric.update(
        {
            "condition_id": _condition_id(stage, preprocess),
            "stage_label": stage.label,
            "stage_name": stage.stage_name,
            "preprocess_condition": preprocess,
            "matcher": local.L2_MNN,
        }
    )
    retrieval = local.retrieval_metrics_for_matchers(
        stage="inner_dev",
        table=retrieval_table,
        tensors=tensors,
        device=device,
        score_batch_size=score_batch_size,
        l3_shift_radius=0,
        methods=[local.L2_MNN],
    )
    retrieval["condition_id"] = _condition_id(stage, preprocess)
    retrieval["stage_label"] = stage.label
    retrieval["stage_name"] = stage.stage_name
    retrieval["preprocess_condition"] = preprocess
    diagnostics = local.aggregate_local_diagnostics(details, stage="inner_dev")
    diagnostics["condition_id"] = _condition_id(stage, preprocess)
    diagnostics["stage_label"] = stage.label
    diagnostics["stage_name"] = stage.stage_name
    diagnostics["preprocess_condition"] = preprocess
    diagnostics["runtime_ms_per_pair"] = metric["runtime_ms_per_pair"]
    return metric, retrieval, diagnostics


def _score_within_controls(
    *,
    stage: StageSpec,
    preprocess: str,
    pair_bundle: dict[str, pd.DataFrame],
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    score_batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in base.WITHIN_PROTOCOLS:
        df = pair_bundle[protocol].reset_index(drop=True).copy()
        df["protocol_id"] = protocol
        scores, _details, elapsed = local.score_pair_frame(
            method=local.L2_MNN,
            df=df,
            tensors=tensors,
            device=device,
            batch_size=score_batch_size,
            l3_shift_radius=0,
        )
        row = local.metric_row_from_scores(
            method=local.L2_MNN,
            stage="inner_dev",
            protocol=protocol,
            df=df,
            scores=scores,
            elapsed_seconds=elapsed,
        )
        row.update(
            {
                "condition_id": _condition_id(stage, preprocess),
                "stage_label": stage.label,
                "stage_name": stage.stage_name,
                "preprocess_condition": preprocess,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_by_preprocess(metrics: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    order = {spec.label: i for i, spec in enumerate(STAGE_SPECS)}
    for preprocess in PREPROCESS_CONDITIONS:
        sub = metrics[metrics["preprocess_condition"] == preprocess].copy()
        sub["_stage_order"] = sub["stage_label"].map(order)
        best = sub.sort_values(["roc_auc", "_stage_order"], ascending=[False, True], kind="mergesort").iloc[0]
        out[preprocess] = {
            "condition_id": str(best["condition_id"]),
            "stage_label": str(best["stage_label"]),
            "stage_name": str(best["stage_name"]),
        }
    return out


def classify_probe(metrics: pd.DataFrame, retrieval: pd.DataFrame, diagnostics: pd.DataFrame, *, cfg: ProbeConfig) -> dict[str, Any]:
    metric_by = {str(row["condition_id"]): row for row in metrics.to_dict("records")}

    def _retrieval_ok(condition_id: str) -> bool:
        rows = retrieval[retrieval["condition_id"] == condition_id]
        by_dir = {str(row["direction"]): row for row in rows.to_dict("records")}
        for direction, baseline in BASELINE_RETRIEVAL.items():
            row = by_dir.get(direction)
            if row is None:
                return False
            if float(row["mrr"]) < float(baseline["mrr"]) + cfg.retrieval_mrr_gain:
                return False
            if float(row["recall_at_1"]) < float(baseline["recall_at_1"]):
                return False
            if float(row["recall_at_5"]) < float(baseline["recall_at_5"]):
                return False
        return True

    def _diagnostics_ok(condition_id: str) -> bool:
        rows = diagnostics[diagnostics["condition_id"] == condition_id]
        gen = rows[rows["pair_group"] == "genuine"]
        imp = rows[rows["pair_group"] == "impostor"]
        if gen.empty or imp.empty:
            return False
        g = gen.iloc[0]
        i = imp.iloc[0]
        cos_gap = float(g["mean_matched_cosine_mean"]) - float(i["mean_matched_cosine_mean"])
        count_gap = float(g["mutual_match_count_mean"]) - float(i["mutual_match_count_mean"])
        return bool(cos_gap >= cfg.diagnostic_cosine_gap and count_gap >= cfg.diagnostic_match_count_gap)

    details: list[dict[str, Any]] = []
    a_conditions: list[str] = []
    for condition_id, row in metric_by.items():
        auc = float(row["roc_auc"])
        auc_gain = auc - BASELINE_F_RAW_L2_AUC
        ret_ok = _retrieval_ok(condition_id)
        diag_ok = _diagnostics_ok(condition_id)
        reaches = auc_gain >= cfg.auc_present_gain and ret_ok and diag_ok
        if reaches:
            a_conditions.append(condition_id)
        details.append(
            {
                "condition_id": condition_id,
                "stage_label": row["stage_label"],
                "preprocess_condition": row["preprocess_condition"],
                "clcb_auc": auc,
                "auc_minus_existing_f_raw_l2": auc_gain,
                "retrieval_consistently_improved": bool(ret_ok),
                "diagnostics_meaningfully_separated": bool(diag_ok),
                "meets_A": bool(reaches),
            }
        )

    p2_gains: list[dict[str, Any]] = []
    improves = 0
    for spec in STAGE_SPECS:
        raw = float(metric_by[f"{spec.label}x{RAW}"]["roc_auc"])
        p2 = float(metric_by[f"{spec.label}x{P2}"]["roc_auc"])
        gain = p2 - raw
        if gain > 0.0:
            improves += 1
        p2_gains.append({"stage_label": spec.label, "p2_minus_raw_auc": gain, "p2_improves": bool(gain > 0.0)})

    best_auc = max(float(row["roc_auc"]) for row in metric_by.values())
    any_retrieval_ok = any(_retrieval_ok(cid) for cid in metric_by)
    all_close = all(abs(float(row["roc_auc"]) - BASELINE_F_RAW_L2_AUC) <= cfg.close_to_baseline_auc for row in metric_by.values())
    diag_nearly_identical = not any(_diagnostics_ok(cid) for cid in metric_by)

    if a_conditions:
        label = "A. FROZEN_LOCAL_SIGNAL_PRESENT"
        reason = "At least one condition passed AUC, retrieval, and local-diagnostic gates."
    elif improves >= cfg.p2_min_stage_improvements and (best_auc < 0.60 or not any_retrieval_ok):
        label = "B. PHOTOMETRIC_HELP_ONLY"
        reason = "P2 improved over RAW for every selected stage, but best AUC stayed below 0.60 or retrieval remained weak."
    elif all_close and not any_retrieval_ok and diag_nearly_identical:
        label = "C. FROZEN_LOCAL_FEATURES_NOT_CROSS_MODAL"
        reason = "All conditions stayed close to the existing F x RAW L2 baseline, with no stable retrieval or diagnostic separation."
    else:
        label = "D. MIXED_OR_INCONCLUSIVE"
        reason = "Layer/preprocessing effects were inconsistent without satisfying A/B/C."

    return {
        "classification": label,
        "reason": reason,
        "criteria": {
            "existing_f_raw_l2_auc_baseline": BASELINE_F_RAW_L2_AUC,
            "auc_present_gain": cfg.auc_present_gain,
            "retrieval_mrr_gain": cfg.retrieval_mrr_gain,
            "diagnostic_cosine_gap": cfg.diagnostic_cosine_gap,
            "diagnostic_match_count_gap": cfg.diagnostic_match_count_gap,
            "p2_min_stage_improvements": cfg.p2_min_stage_improvements,
            "close_to_baseline_auc": cfg.close_to_baseline_auc,
            "official_val_policy": "closed in Phase 4B.2A.1 regardless of A/B/C/D",
        },
        "condition_details": details,
        "p2_stage_gains": p2_gains,
        "official_val": {"opened": False, "reason": "Official VAL is explicitly closed for Phase 4B.2A.1."},
    }


def _experiment_config(cfg: ProbeConfig, *, checkpoint: Path, phase4b1_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_required": local.DEFAULT_CHECKPOINT_SHA256,
        "phase4b1_inner_split_source": str(Path(phase4b1_dir) / "inner_split.json"),
        "stage_selection": [spec.__dict__ for spec in STAGE_SPECS],
        "descriptor_grid": {
            "target_grid": [GRID_SIZE, GRID_SIZE],
            "pool_if_native_larger": "adaptive_avg_pool2d((24,24))",
            "preserve_if_native_24x24": True,
            "upsample_lower_resolution": False,
            "max_descriptors": GRID_SIZE * GRID_SIZE,
            "descriptor_normalization": "L2 per local descriptor after pooling/preserve operation",
        },
        "preprocessing_conditions": {
            RAW: base.PREPROCESS_CONTRACT,
            P2: {
                **base.PREPROCESS_CONTRACT,
                "contactless_only_p2_robust_intensity_norm": {
                    "lo_pct": P2_LO_PCT,
                    "hi_pct": P2_HI_PCT,
                    "map": "p05->0, p95->255, clip",
                    "degenerate": "range<1 -> identity",
                },
                "contact_based_unchanged": True,
            },
        },
        "matcher": {
            "name": local.L2_MNN,
            "score_semantics": "coverage * mean mutual-nearest-neighbor cosine; coverage=count/min(num_a,num_b); no matches score -1",
        },
        "matrix": [f"{spec.label}x{prep}" for spec in STAGE_SPECS for prep in PREPROCESS_CONDITIONS],
        "decision_logic": {
            "A": "AUC >= existing F x RAW L2 + 0.05, retrieval improves both directions, diagnostics separate.",
            "B": "P2 improves over RAW for every selected stage, but best AUC <0.60 or retrieval weak.",
            "C": "All conditions close to baseline, no stable retrieval, diagnostics nearly identical.",
            "D": "Mixed or inconsistent effects.",
        },
        "training": {"performed": False, "optimizer_instantiated": False, "encoder_frozen": True},
        "official_val_policy": "closed",
        "test_policy": "TEST is never read",
    }


def run(
    *,
    manifest_dir: Path,
    controls_dir: Path,
    phase4b1_dir: Path,
    checkpoint: Path,
    outdir: Path,
    polyu_root: Optional[str],
    device_arg: str,
    cfg: ProbeConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = local.resolve_device(device_arg)
    checkpoint = Path(checkpoint)
    write_json(outdir / "experiment_config.json", _experiment_config(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir))

    train_images_all, resolved_root = local.load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[base.TRAIN])
    train_ids = sorted(train_images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = local.load_fixed_inner_split(train_ids, phase4b1_dir, split_seed=cfg.seed)
    inner_dev_ids = inner_split["inner_dev"]
    inner_dev_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    if smoke:
        keep_ids = inner_dev_ids[: min(8, len(inner_dev_ids))]
        inner_dev_ids = keep_ids
        inner_dev_images = inner_dev_images[inner_dev_images["finger_unit_id"].astype(str).isin(set(keep_ids))].copy()

    pair_max_pos = min(cfg.eval_max_pos, 40) if smoke else cfg.eval_max_pos
    pair_neg_per_pos = min(cfg.eval_neg_per_pos, 1) if smoke else cfg.eval_neg_per_pos
    pair_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=pair_max_pos,
        neg_per_pos=pair_neg_per_pos,
        seed=cfg.seed,
    )
    inner_counts = base.validate_pair_bundle(pair_bundle, stage="inner_dev")
    retrieval_table = base.build_retrieval_table(inner_dev_images, inner_dev_ids)

    model, ckpt_args, checkpoint_meta, checkpoint_sha = local.load_frozen_pair_model(checkpoint=checkpoint, device=device)
    input_size = int(ckpt_args.get("input_size", 384))
    width = int(ckpt_args.get("width", 32))
    stage_inventory = feature_stage_inventory(width=width, input_size=input_size)
    write_csv(outdir / "feature_stage_inventory.csv", stage_inventory)

    uids = local._unique_uids_for_pairs(pair_bundle, inner_dev_images)
    image_stores = {
        preprocess: load_image_store_for_condition(inner_dev_images, input_size=input_size, preprocess=preprocess)
        for preprocess in PREPROCESS_CONDITIONS
    }

    metric_rows: list[dict[str, Any]] = []
    retrieval_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    cache_meta: dict[str, Any] = {}
    for stage in STAGE_SPECS:
        for preprocess in PREPROCESS_CONDITIONS:
            condition_id = _condition_id(stage, preprocess)
            print(f"[condition] {condition_id}", flush=True)
            cache, meta = extract_stage_descriptor_cache(
                encoder=model.encoder,
                image_store=image_stores[preprocess],
                uids=uids,
                device=device,
                batch_size=cfg.eval_batch_size,
                stage=stage,
                amp=cfg.amp,
            )
            cache_meta[condition_id] = meta
            tensors = local.descriptor_cache_to_tensors(cache, device)
            metric, retrieval, diagnostics = _score_clcb_condition(
                stage=stage,
                preprocess=preprocess,
                pair_df=pair_bundle["contactless_to_contact_based"],
                retrieval_table=retrieval_table,
                tensors=tensors,
                device=device,
                score_batch_size=cfg.score_batch_size,
            )
            metric.update(meta)
            metric_rows.append(metric)
            retrieval_frames.append(retrieval)
            diagnostic_frames.append(diagnostics)
            del tensors, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    retrieval_df = pd.concat(retrieval_frames, ignore_index=True) if retrieval_frames else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    decision = classify_probe(metrics, retrieval_df, diagnostics_df, cfg=cfg)
    best_by_preprocess = select_best_by_preprocess(metrics)

    within_frames: list[pd.DataFrame] = []
    for preprocess, selected in best_by_preprocess.items():
        stage = next(spec for spec in STAGE_SPECS if spec.label == selected["stage_label"])
        condition_id = _condition_id(stage, preprocess)
        print(f"[within] {condition_id}", flush=True)
        cache, _meta = extract_stage_descriptor_cache(
            encoder=model.encoder,
            image_store=image_stores[preprocess],
            uids=uids,
            device=device,
            batch_size=cfg.eval_batch_size,
            stage=stage,
            amp=cfg.amp,
        )
        tensors = local.descriptor_cache_to_tensors(cache, device)
        within_frames.append(
            _score_within_controls(
                stage=stage,
                preprocess=preprocess,
                pair_bundle=pair_bundle,
                tensors=tensors,
                device=device,
                score_batch_size=cfg.score_batch_size,
            )
        )
        del tensors, cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    within_df = pd.concat(within_frames, ignore_index=True) if within_frames else pd.DataFrame()

    write_csv(outdir / "layer_p2_metrics.csv", metrics)
    write_csv(outdir / "retrieval_metrics.csv", retrieval_df)
    write_csv(outdir / "local_diagnostics.csv", diagnostics_df)
    write_csv(outdir / "within_modality_controls.csv", within_df)
    write_json(outdir / "decision.json", decision)

    canonical_files = {
        "manifest_csv": Path(manifest_dir) / "manifest.csv",
        "pairs_train_csv": Path(manifest_dir) / "pairs_train.csv",
        "checkpoint": checkpoint,
    }
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "dataset": base.DATASET_NAME,
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_args": ckpt_args,
        "preprocess_conditions": list(PREPROCESS_CONDITIONS),
        "feature_stages": stage_inventory.to_dict("records"),
        "descriptor_cache_meta": cache_meta,
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "inner_split": {
            "source": str(Path(phase4b1_dir) / "inner_split.json"),
            "official_train_identity_count": int(len(train_ids)),
            "inner_dev_identity_count": int(len(inner_dev_ids)),
            "smoke": bool(smoke),
        },
        "pair_counts": {"inner_dev": inner_counts},
        "best_by_preprocess_for_within_controls": best_by_preprocess,
        "decision": decision,
        "outputs": {
            "feature_stage_inventory_csv": str(outdir / "feature_stage_inventory.csv"),
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "layer_p2_metrics_csv": str(outdir / "layer_p2_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "local_diagnostics_csv": str(outdir / "local_diagnostics.csv"),
            "within_modality_controls_csv": str(outdir / "within_modality_controls.csv"),
            "decision_json": str(outdir / "decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
        },
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
        "canonical_artifacts_not_read": {
            "pairs_val_csv": str(Path(manifest_dir) / "pairs_val.csv"),
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "Official VAL and TEST remain closed for Phase 4B.2A.1.",
        },
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {
            "numpy": safe_pkg_version("numpy"),
            "pandas": safe_pkg_version("pandas"),
            "torch": safe_pkg_version("torch"),
            "scikit-learn": safe_pkg_version("scikit-learn"),
        },
        "constraints": {
            "primary_data": "TRAIN inner-dev identities only",
            "official_val_read": False,
            "test_pairs_read": False,
            "test_images_loaded": False,
            "canonical_manifest_or_pairs_modified": False,
            "canonical_checkpoint_modified": False,
            "encoder_frozen": True,
            "optimizer_instantiated": False,
            "training_performed": False,
            "used_fusion": False,
            "used_transformer_or_cross_attention": False,
            "used_learned_alignment_or_unwarping": False,
            "broad_layer_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "metrics": metrics,
        "retrieval": retrieval_df,
        "diagnostics": diagnostics_df,
        "within": within_df,
        "decision": decision,
        "run_manifest": run_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.2A.1 bounded Layer x P2 local-signal probe.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=ProbeConfig.seed)
    p.add_argument("--eval_max_pos", type=int, default=ProbeConfig.eval_max_pos)
    p.add_argument("--eval_neg_per_pos", type=int, default=ProbeConfig.eval_neg_per_pos)
    p.add_argument("--eval_batch_size", type=int, default=ProbeConfig.eval_batch_size)
    p.add_argument("--score_batch_size", type=int, default=ProbeConfig.score_batch_size)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ProbeConfig(
        seed=int(args.seed),
        eval_max_pos=int(args.eval_max_pos),
        eval_neg_per_pos=int(args.eval_neg_per_pos),
        eval_batch_size=int(args.eval_batch_size),
        score_batch_size=int(args.score_batch_size),
        amp=not bool(args.no_amp),
    )
    try:
        result = run(
            manifest_dir=resolve_repo_path(args.data_dir),
            controls_dir=resolve_repo_path(args.controls_dir),
            phase4b1_dir=resolve_repo_path(args.phase4b1_dir),
            checkpoint=resolve_repo_path(args.checkpoint),
            outdir=resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            device_arg=str(args.device),
            cfg=cfg,
            smoke=bool(args.smoke),
        )
    except (LayerP2ProbeError, local.LocalCorrespondenceError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metrics = result["metrics"]
    print("\n=== PolyU Cross Phase 4B.2A.1 Layer x P2 probe complete ===")
    print(f"Output dir     : {result['outdir']}")
    print(f"Classification : {result['decision']['classification']}")
    print("\nInner-dev CL->CB L2 MNN AUC:")
    print(metrics[["condition_id", "stage_name", "preprocess_condition", "roc_auc", "eer", "runtime_ms_per_pair"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
