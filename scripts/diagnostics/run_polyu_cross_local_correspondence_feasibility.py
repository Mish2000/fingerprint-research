"""Phase 4B.2A local-correspondence feasibility audit for PolyU Cross.

This diagnostic asks whether the frozen canonical SD300 ConvEncoder contains
useful local cross-modal correspondence signal that global average pooling and
global cosine similarity lose.

Guardrails:
* Primary audit uses only the frozen Phase 4B.1 TRAIN inner-dev identity split.
* Official VAL is opened only after the predeclared TRAIN/inner-dev gate.
* TEST pairs/images are never read.
* The canonical checkpoint, manifest, and pair bundles are read-only.
* No optimizer is instantiated, and no encoder/projection training is run.
* No transformer, cross-attention, Ridgeformer, Fusion, P2 primary input, or
  learned alignment is introduced.
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
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, safe_pkg_version, sha256_file, utc_now
from scripts.deep.score_fast_pair_ddp_splits import PairModel, load_image_u8
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import (
    PolyUCrossPairError,
    resolve_pair_image_path,
    resolve_polyu_cross_root,
)


RUN_SCHEMA_VERSION = "polyu_cross_local_correspondence_feasibility_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_local_correspondence_feasibility_v0"
DEFAULT_CHECKPOINT_SHA256 = "0541c16a3e0c05638cfda2a6ccb928d0fb86988cc5f76e9524d70cfa1640584e"

TRAIN = base.TRAIN
VAL = base.VAL
CONTACT = base.CONTACT
CONTACTLESS = base.CONTACTLESS
CONTROL_PROTOCOLS = base.CONTROL_PROTOCOLS
WITHIN_PROTOCOLS = base.WITHIN_PROTOCOLS

L0_GLOBAL = "L0_global_cosine"
L1_CHAMFER = "L1_symmetric_local_chamfer"
L2_MNN = "L2_mutual_nearest_neighbor_local"
L3_SPATIAL = "L3_coarse_spatial_local_matching"
LOCAL_METHODS = (L1_CHAMFER, L2_MNN, L3_SPATIAL)
ALL_METHODS = (L0_GLOBAL, *LOCAL_METHODS)
METHOD_SIMPLICITY_RANK = {
    L1_CHAMFER: 1,
    L2_MNN: 2,
    L3_SPATIAL: 3,
}

SELECTED_STAGE_INDEX = 3
SELECTED_STAGE_NAME = "encoder.net.3"
PREDECLARED_SHIFT_RADIUS = 2


class LocalCorrespondenceError(RuntimeError):
    """Raised for Phase 4B.2A protocol or artifact errors."""


@dataclass(frozen=True)
class LocalAuditConfig:
    seed: int = 1341
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 64
    score_batch_size: int = 16
    selected_stage_index: int = SELECTED_STAGE_INDEX
    l3_shift_radius: int = PREDECLARED_SHIFT_RADIUS
    amp: bool = True
    material_auc_gain: float = 0.03
    close_auc_margin: float = 0.02
    retrieval_mrr_gain: float = 0.03
    retrieval_close_margin: float = 0.02
    within_auc_drop_tolerance: float = 0.05
    local_selection_auc_margin: float = 0.02
    descriptor_disk_cache: bool = True


@dataclass
class DescriptorCache:
    sample_uids: list[str]
    uid_to_index: dict[str, int]
    local_grids: dict[str, np.ndarray]
    global_embeddings: dict[str, np.ndarray]
    height: int
    width: int
    channels: int
    selected_stage_name: str


@dataclass
class TensorDescriptorCache:
    local_flat: torch.Tensor
    local_grid: torch.Tensor
    global_embeddings: torch.Tensor
    uid_to_index: dict[str, int]


@dataclass(frozen=True)
class PairScore:
    score: float
    failed: bool = False
    mutual_match_count: int = 0
    mean_matched_cosine: float = float("nan")
    median_matched_cosine: float = float("nan")
    chamfer_a_to_b: float = float("nan")
    chamfer_b_to_a: float = float("nan")
    best_dx: float = float("nan")
    best_dy: float = float("nan")
    best_overlap_count: int = 0
    best_aligned_cosine: float = float("nan")


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def resolve_device(raw: str) -> torch.device:
    raw = str(raw).strip().lower()
    if raw in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def load_manifest_for_splits(
    manifest_dir: Path,
    *,
    polyu_root: Optional[str],
    splits: Iterable[str],
) -> tuple[pd.DataFrame, Any]:
    wanted = {str(s) for s in splits}
    if "test" in wanted:
        raise LocalCorrespondenceError("TEST split is prohibited for Phase 4B.2A")

    manifest_csv = Path(manifest_dir) / "manifest.csv"
    if not manifest_csv.exists():
        raise LocalCorrespondenceError(f"Missing PolyU Cross manifest: {manifest_csv}")
    required = {"finger_unit_id", "sample_uid", "modality", "session_id", "split", "path"}
    frame = pd.read_csv(manifest_csv, dtype=str)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LocalCorrespondenceError(f"manifest.csv missing columns {missing}; found {list(frame.columns)}")

    frame = frame[frame["split"].isin(sorted(wanted))].copy()
    frame = frame[
        ["finger_unit_id", "sample_uid", "modality", "session_id", "split", "path"]
    ].sort_values(["split", "finger_unit_id", "modality", "session_id", "sample_uid"], kind="mergesort")
    if frame.empty:
        raise LocalCorrespondenceError(f"No rows found for requested split(s): {sorted(wanted)}")

    resolved_root = resolve_polyu_cross_root(Path(manifest_dir), override=polyu_root)
    resolved_paths: list[str] = []
    exists: list[bool] = []
    for raw in frame["path"].astype(str):
        path = resolve_pair_image_path(raw, resolved_root.root)
        resolved_paths.append(str(path))
        exists.append(path.exists())
    frame["resolved_path"] = resolved_paths
    frame["path_exists"] = exists
    missing_rows = frame[~frame["path_exists"]]
    if not missing_rows.empty:
        first = missing_rows.iloc[0]
        raise LocalCorrespondenceError(
            f"{len(missing_rows)} requested manifest image(s) are missing; first={first['resolved_path']!r}"
        )
    return frame.reset_index(drop=True), resolved_root


def load_fixed_inner_split(train_ids: list[str], phase4b1_dir: Path, *, split_seed: int) -> dict[str, list[str]]:
    path = Path(phase4b1_dir) / "inner_split.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        inner = {
            "inner_train": [str(x) for x in payload.get("inner_train", [])],
            "inner_dev": [str(x) for x in payload.get("inner_dev", [])],
        }
    else:
        inner = base.make_inner_split(train_ids, dev_fraction=0.15, seed=int(split_seed))
    if not inner["inner_train"] or not inner["inner_dev"]:
        raise LocalCorrespondenceError("Inner split is empty")
    if set(inner["inner_train"]).intersection(inner["inner_dev"]):
        raise LocalCorrespondenceError("Inner split is not identity-disjoint")
    if set(inner["inner_train"]).union(inner["inner_dev"]) != set(map(str, train_ids)):
        raise LocalCorrespondenceError("Inner split does not match official TRAIN identities")
    return {
        "inner_train": sorted(inner["inner_train"], key=base.natural_identity_key),
        "inner_dev": sorted(inner["inner_dev"], key=base.natural_identity_key),
    }


def convencoder_feature_map_inventory(
    *,
    width: int,
    input_size: int,
    selected_stage_index: int = SELECTED_STAGE_INDEX,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rf = 1
    stride = 1
    spatial = int(input_size)
    channels_by_stage = [int(width), int(width) * 2, int(width) * 4, int(width) * 8]
    for stage_index, channels in enumerate(channels_by_stage):
        rf += 2 * stride  # first 3x3 conv
        rf += 2 * stride  # second 3x3 conv
        rf += stride  # 2x2 max-pool
        stride *= 2
        spatial //= 2
        rows.append(
            {
                "stage_name": f"encoder.net.{stage_index}",
                "module_type": "conv_block_with_final_maxpool",
                "spatial_height": int(spatial),
                "spatial_width": int(spatial),
                "channel_count": int(channels),
                "local_descriptor_dim": int(channels),
                "receptive_field_pixels": int(rf),
                "effective_stride_pixels": int(stride),
                "receptive_field_note": (
                    f"Approximate ConvEncoder RF after two 3x3 stride-1 convs and one 2x2 pool: {rf}px; "
                    f"grid stride {stride}px."
                ),
                "before_or_after_final_pooling": "before_final_adaptive_avg_pool",
                "selected_for_local_descriptors": bool(stage_index == int(selected_stage_index)),
            }
        )
    rows.append(
        {
            "stage_name": "encoder.net.4",
            "module_type": "adaptive_avg_pool2d",
            "spatial_height": 1,
            "spatial_width": 1,
            "channel_count": int(width) * 8,
            "local_descriptor_dim": int(width) * 8,
            "receptive_field_pixels": "",
            "effective_stride_pixels": "",
            "receptive_field_note": "Global average over the final 24x24 feature grid; spatial correspondence is collapsed.",
            "before_or_after_final_pooling": "after_final_adaptive_avg_pool",
            "selected_for_local_descriptors": False,
        }
    )
    return rows


def load_frozen_pair_model(
    *,
    checkpoint: Path,
    device: torch.device,
    require_sha256: str = DEFAULT_CHECKPOINT_SHA256,
) -> tuple[PairModel, dict[str, Any], dict[str, Any], str]:
    checkpoint = Path(checkpoint)
    checkpoint_sha = sha256_file(checkpoint) or ""
    if require_sha256 and checkpoint_sha != require_sha256:
        raise LocalCorrespondenceError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {require_sha256}")
    payload, meta = base.load_checkpoint_payload(checkpoint)
    args = dict(payload.get("args", {}) or {})
    width = int(args.get("width", 32))
    embedding_dim = int(args.get("embedding_dim", 512))
    hidden_dim = int(args.get("hidden_dim", 768))
    model = PairModel(width=width, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, args, meta, checkpoint_sha


def _target_stage(encoder: nn.Module, stage_index: int) -> nn.Module:
    if not hasattr(encoder, "net"):
        raise LocalCorrespondenceError("ConvEncoder is expected to expose encoder.net")
    net = getattr(encoder, "net")
    if int(stage_index) < 0 or int(stage_index) >= len(net):
        raise LocalCorrespondenceError(f"Invalid encoder stage index: {stage_index}")
    return net[int(stage_index)]


@torch.inference_mode()
def extract_descriptor_cache(
    *,
    encoder: nn.Module,
    image_store: dict[str, torch.Tensor],
    uids: list[str],
    device: torch.device,
    batch_size: int,
    selected_stage_index: int,
    amp: bool,
) -> DescriptorCache:
    encoder.eval()
    target = _target_stage(encoder, selected_stage_index)
    selected_stage_name = f"encoder.net.{int(selected_stage_index)}"
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = target.register_forward_hook(_hook)
    local_grids: dict[str, np.ndarray] = {}
    global_embeddings: dict[str, np.ndarray] = {}
    height = width = channels = 0
    try:
        for start in range(0, len(uids), int(batch_size)):
            chunk = uids[start : start + int(batch_size)]
            captured.clear()
            x = base.stack_batch(image_store, chunk, device=device)
            with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
                emb = encoder(x)
            if len(captured) != 1:
                raise LocalCorrespondenceError(
                    f"Expected exactly one captured feature map from {selected_stage_name}; got {len(captured)}"
                )
            fmap = captured[0].float()
            if fmap.ndim != 4:
                raise LocalCorrespondenceError(f"Selected feature map is not spatial: shape={tuple(fmap.shape)}")
            local = fmap.permute(0, 2, 3, 1).contiguous()
            local = F.normalize(local, p=2, dim=3)
            global_z = F.normalize(emb.float(), p=2, dim=1)
            h, w, c = int(local.shape[1]), int(local.shape[2]), int(local.shape[3])
            if height == 0:
                height, width, channels = h, w, c
            elif (height, width, channels) != (h, w, c):
                raise LocalCorrespondenceError("Feature-map dimensions changed across batches")
            local_np = local.detach().cpu().numpy().astype(np.float32, copy=True)
            global_np = global_z.detach().cpu().numpy().astype(np.float32, copy=True)
            for uid, grid, vec in zip(chunk, local_np, global_np):
                local_grids[str(uid)] = grid
                global_embeddings[str(uid)] = vec
    finally:
        handle.remove()
    ordered = [str(uid) for uid in uids]
    return DescriptorCache(
        sample_uids=ordered,
        uid_to_index={uid: i for i, uid in enumerate(ordered)},
        local_grids=local_grids,
        global_embeddings=global_embeddings,
        height=height,
        width=width,
        channels=channels,
        selected_stage_name=selected_stage_name,
    )


def descriptor_cache_to_tensors(cache: DescriptorCache, device: torch.device) -> TensorDescriptorCache:
    grid = np.stack([cache.local_grids[uid] for uid in cache.sample_uids], axis=0).astype(np.float32, copy=False)
    global_embeddings = np.stack([cache.global_embeddings[uid] for uid in cache.sample_uids], axis=0).astype(
        np.float32, copy=False
    )
    local_grid = torch.as_tensor(grid, dtype=torch.float32, device=device)
    local_flat = local_grid.reshape(local_grid.shape[0], local_grid.shape[1] * local_grid.shape[2], local_grid.shape[3])
    global_tensor = torch.as_tensor(global_embeddings, dtype=torch.float32, device=device)
    return TensorDescriptorCache(
        local_flat=local_flat,
        local_grid=local_grid,
        global_embeddings=global_tensor,
        uid_to_index=cache.uid_to_index,
    )


def write_descriptor_cache_artifacts(outdir: Path, cache: DescriptorCache, images: pd.DataFrame) -> dict[str, str]:
    cache_dir = Path(outdir) / "descriptor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered = cache.sample_uids
    local = np.stack([cache.local_grids[uid] for uid in ordered], axis=0).astype(np.float16)
    global_embeddings = np.stack([cache.global_embeddings[uid] for uid in ordered], axis=0).astype(np.float32)
    npz_path = cache_dir / "spatial_descriptors_stage_encoder_net_3_float16.npz"
    np.savez_compressed(
        npz_path,
        sample_uid=np.asarray(ordered, dtype=str),
        local_descriptor_grid=local,
        global_embedding=global_embeddings,
        height=np.asarray([cache.height], dtype=np.int32),
        width=np.asarray([cache.width], dtype=np.int32),
        channels=np.asarray([cache.channels], dtype=np.int32),
    )
    meta = images.drop_duplicates("sample_uid").copy()
    meta = meta[meta["sample_uid"].astype(str).isin(set(ordered))].copy()
    order = {uid: i for i, uid in enumerate(ordered)}
    meta["cache_order"] = meta["sample_uid"].astype(str).map(order)
    meta = meta.sort_values("cache_order", kind="mergesort")
    manifest_csv = cache_dir / "descriptor_cache_manifest.csv"
    write_csv(
        manifest_csv,
        meta[
            [
                "cache_order",
                "sample_uid",
                "finger_unit_id",
                "modality",
                "session_id",
                "split",
                "path",
                "resolved_path",
            ]
        ],
    )
    return {
        "descriptor_npz": str(npz_path),
        "descriptor_cache_manifest_csv": str(manifest_csv),
        "storage_dtype": "float16_for_spatial_grid_float32_for_global_embedding",
    }


def local_flat(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected local descriptor grid/flat matrix, got shape={arr.shape}")


def symmetric_local_chamfer_np(a_grid: np.ndarray, b_grid: np.ndarray) -> PairScore:
    a = local_flat(a_grid)
    b = local_flat(b_grid)
    if a.size == 0 or b.size == 0:
        return PairScore(score=float("nan"), failed=True)
    sim = a @ b.T
    a_to_b = np.max(sim, axis=1)
    b_to_a = np.max(sim, axis=0)
    score = 0.5 * (float(np.mean(a_to_b)) + float(np.mean(b_to_a)))
    return PairScore(score=score, chamfer_a_to_b=float(np.mean(a_to_b)), chamfer_b_to_a=float(np.mean(b_to_a)))


def mutual_nearest_neighbor_np(a_grid: np.ndarray, b_grid: np.ndarray) -> PairScore:
    a = local_flat(a_grid)
    b = local_flat(b_grid)
    if a.size == 0 or b.size == 0:
        return PairScore(score=float("nan"), failed=True)
    sim = a @ b.T
    nn_ab = np.argmax(sim, axis=1)
    nn_ba = np.argmax(sim, axis=0)
    source = np.arange(a.shape[0])
    mutual = nn_ba[nn_ab] == source
    mutual_i = source[mutual]
    mutual_j = nn_ab[mutual]
    if mutual_i.size == 0:
        return PairScore(score=-1.0, mutual_match_count=0)
    values = sim[mutual_i, mutual_j]
    coverage = float(mutual_i.size / max(1, min(a.shape[0], b.shape[0])))
    mean_cos = float(np.mean(values))
    return PairScore(
        score=float(coverage * mean_cos),
        mutual_match_count=int(mutual_i.size),
        mean_matched_cosine=mean_cos,
        median_matched_cosine=float(np.median(values)),
    )


def coarse_offsets(radius: int) -> list[tuple[int, int]]:
    r = int(radius)
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return sorted(offsets, key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1]), item[0], item[1]))


def _overlap_slices(height: int, width: int, dy: int, dx: int) -> tuple[slice, slice, slice, slice]:
    if dy >= 0:
        ay = slice(0, height - dy)
        by = slice(dy, height)
    else:
        ay = slice(-dy, height)
        by = slice(0, height + dy)
    if dx >= 0:
        ax = slice(0, width - dx)
        bx = slice(dx, width)
    else:
        ax = slice(-dx, width)
        bx = slice(0, width + dx)
    return ay, ax, by, bx


def coarse_spatial_match_np(a_grid: np.ndarray, b_grid: np.ndarray, *, radius: int = PREDECLARED_SHIFT_RADIUS) -> PairScore:
    a = np.asarray(a_grid, dtype=np.float32)
    b = np.asarray(b_grid, dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3 or a.shape[2] != b.shape[2]:
        return PairScore(score=float("nan"), failed=True)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    best = -math.inf
    best_dx = 0
    best_dy = 0
    best_overlap = 0
    for dy, dx in coarse_offsets(radius):
        if abs(dy) >= h or abs(dx) >= w:
            continue
        ay, ax, by, bx = _overlap_slices(h, w, dy, dx)
        sim = np.sum(a[ay, ax, :] * b[by, bx, :], axis=2)
        if sim.size == 0:
            continue
        score = float(np.mean(sim))
        if score > best + 1e-12:
            best = score
            best_dx = int(dx)
            best_dy = int(dy)
            best_overlap = int(sim.size)
    if not math.isfinite(best):
        return PairScore(score=float("nan"), failed=True)
    return PairScore(
        score=float(best),
        best_dx=float(best_dx),
        best_dy=float(best_dy),
        best_overlap_count=int(best_overlap),
        best_aligned_cosine=float(best),
    )


def _pair_indices(df: pd.DataFrame, uid_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_a = np.full(len(df), -1, dtype=np.int64)
    idx_b = np.full(len(df), -1, dtype=np.int64)
    valid = np.ones(len(df), dtype=bool)
    for i, row in enumerate(df.itertuples(index=False)):
        uid_a = str(getattr(row, "sample_uid_a"))
        uid_b = str(getattr(row, "sample_uid_b"))
        if uid_a not in uid_to_index or uid_b not in uid_to_index:
            valid[i] = False
            continue
        idx_a[i] = uid_to_index[uid_a]
        idx_b[i] = uid_to_index[uid_b]
    return idx_a, idx_b, valid


@torch.inference_mode()
def score_pair_frame(
    *,
    method: str,
    df: pd.DataFrame,
    tensors: TensorDescriptorCache,
    device: torch.device,
    batch_size: int,
    l3_shift_radius: int,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    scores = np.full(len(df), np.nan, dtype=np.float64)
    detail_rows: list[dict[str, Any]] = []
    idx_a, idx_b, valid = _pair_indices(df, tensors.uid_to_index)
    valid_positions = np.flatnonzero(valid)
    start_time = time.perf_counter()
    for start in range(0, len(valid_positions), int(batch_size)):
        pos = valid_positions[start : start + int(batch_size)]
        a_idx = torch.as_tensor(idx_a[pos], dtype=torch.long, device=device)
        b_idx = torch.as_tensor(idx_b[pos], dtype=torch.long, device=device)
        if method == L0_GLOBAL:
            a = tensors.global_embeddings.index_select(0, a_idx)
            b = tensors.global_embeddings.index_select(0, b_idx)
            batch_scores = torch.sum(a * b, dim=1).detach().cpu().numpy()
            scores[pos] = batch_scores
            for row_pos, score in zip(pos, batch_scores):
                detail_rows.append(_detail_row(df.iloc[int(row_pos)], method, PairScore(score=float(score))))
        elif method in (L1_CHAMFER, L2_MNN):
            a = tensors.local_flat.index_select(0, a_idx)
            b = tensors.local_flat.index_select(0, b_idx)
            sim = torch.bmm(a, b.transpose(1, 2))
            if method == L1_CHAMFER:
                a_to_b = torch.max(sim, dim=2).values
                b_to_a = torch.max(sim, dim=1).values
                batch_scores_t = 0.5 * (torch.mean(a_to_b, dim=1) + torch.mean(b_to_a, dim=1))
                batch_scores = batch_scores_t.detach().cpu().numpy()
                scores[pos] = batch_scores
                a_mean = torch.mean(a_to_b, dim=1).detach().cpu().numpy()
                b_mean = torch.mean(b_to_a, dim=1).detach().cpu().numpy()
                for row_pos, score, ab, ba in zip(pos, batch_scores, a_mean, b_mean):
                    detail_rows.append(
                        _detail_row(
                            df.iloc[int(row_pos)],
                            method,
                            PairScore(score=float(score), chamfer_a_to_b=float(ab), chamfer_b_to_a=float(ba)),
                        )
                    )
            else:
                nn_ab = torch.argmax(sim, dim=2)
                nn_ba = torch.argmax(sim, dim=1)
                chosen = torch.gather(nn_ba, 1, nn_ab)
                source = torch.arange(sim.shape[1], device=device).view(1, -1).expand_as(chosen)
                mutual = chosen == source
                nn_scores = torch.gather(sim, 2, nn_ab.unsqueeze(2)).squeeze(2)
                mutual_cpu = mutual.detach().cpu().numpy()
                nn_scores_cpu = nn_scores.detach().cpu().numpy()
                for j, row_pos in enumerate(pos):
                    values = nn_scores_cpu[j][mutual_cpu[j]]
                    if values.size == 0:
                        pair_score = PairScore(score=-1.0, mutual_match_count=0)
                    else:
                        coverage = float(values.size / max(1, min(sim.shape[1], sim.shape[2])))
                        mean_cos = float(np.mean(values))
                        pair_score = PairScore(
                            score=float(coverage * mean_cos),
                            mutual_match_count=int(values.size),
                            mean_matched_cosine=mean_cos,
                            median_matched_cosine=float(np.median(values)),
                        )
                    scores[int(row_pos)] = pair_score.score
                    detail_rows.append(_detail_row(df.iloc[int(row_pos)], method, pair_score))
        elif method == L3_SPATIAL:
            a = tensors.local_grid.index_select(0, a_idx)
            b = tensors.local_grid.index_select(0, b_idx)
            batch_n, h, w, _c = a.shape
            best = torch.full((batch_n,), -float("inf"), dtype=torch.float32, device=device)
            best_dx = torch.zeros((batch_n,), dtype=torch.int64, device=device)
            best_dy = torch.zeros((batch_n,), dtype=torch.int64, device=device)
            best_overlap = torch.zeros((batch_n,), dtype=torch.int64, device=device)
            for dy, dx in coarse_offsets(l3_shift_radius):
                if abs(dy) >= h or abs(dx) >= w:
                    continue
                ay, ax, by, bx = _overlap_slices(h, w, dy, dx)
                sim = torch.sum(a[:, ay, ax, :] * b[:, by, bx, :], dim=3)
                candidate = torch.mean(sim.reshape(batch_n, -1), dim=1)
                update = candidate > best + 1e-12
                best = torch.where(update, candidate, best)
                best_dx = torch.where(update, torch.full_like(best_dx, int(dx)), best_dx)
                best_dy = torch.where(update, torch.full_like(best_dy, int(dy)), best_dy)
                best_overlap = torch.where(update, torch.full_like(best_overlap, int(sim.shape[1] * sim.shape[2])), best_overlap)
            batch_scores = best.detach().cpu().numpy()
            scores[pos] = batch_scores
            dx_np = best_dx.detach().cpu().numpy()
            dy_np = best_dy.detach().cpu().numpy()
            ov_np = best_overlap.detach().cpu().numpy()
            for row_pos, score, dx, dy, overlap in zip(pos, batch_scores, dx_np, dy_np, ov_np):
                pair_score = PairScore(
                    score=float(score),
                    best_dx=float(dx),
                    best_dy=float(dy),
                    best_overlap_count=int(overlap),
                    best_aligned_cosine=float(score),
                )
                detail_rows.append(_detail_row(df.iloc[int(row_pos)], method, pair_score))
        else:
            raise LocalCorrespondenceError(f"Unknown matcher: {method}")
    elapsed = time.perf_counter() - start_time
    invalid_positions = np.flatnonzero(~valid)
    for row_pos in invalid_positions:
        detail_rows.append(_detail_row(df.iloc[int(row_pos)], method, PairScore(score=float("nan"), failed=True)))
    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail = detail.sort_values(["_row_order"], kind="mergesort").drop(columns=["_row_order"])
    return scores, detail, float(elapsed)


def _detail_row(row: pd.Series, method: str, pair_score: PairScore) -> dict[str, Any]:
    return {
        "_row_order": int(row.name) if row.name is not None else 0,
        "method": method,
        "protocol": str(row.get("protocol_id", "")),
        "pair_id": str(row.get("pair_id", "")),
        "label": int(row.get("label", -1)),
        "sample_uid_a": str(row.get("sample_uid_a", "")),
        "sample_uid_b": str(row.get("sample_uid_b", "")),
        "finger_unit_a": str(row.get("finger_unit_a", "")),
        "finger_unit_b": str(row.get("finger_unit_b", "")),
        "score": pair_score.score,
        "failed": bool(pair_score.failed or not math.isfinite(float(pair_score.score))),
        "mutual_match_count": int(pair_score.mutual_match_count),
        "mean_matched_cosine": pair_score.mean_matched_cosine,
        "median_matched_cosine": pair_score.median_matched_cosine,
        "chamfer_a_to_b": pair_score.chamfer_a_to_b,
        "chamfer_b_to_a": pair_score.chamfer_b_to_a,
        "best_dx": pair_score.best_dx,
        "best_dy": pair_score.best_dy,
        "best_overlap_count": int(pair_score.best_overlap_count),
        "best_aligned_cosine": pair_score.best_aligned_cosine,
    }


def metric_row_from_scores(
    *,
    method: str,
    stage: str,
    protocol: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    labels = df["label"].astype(int).to_numpy()
    auc, eer = base.auc_eer(labels, scores)
    gen_stats = base.group_stats(scores[labels == 1])
    imp_stats = base.group_stats(scores[labels == 0])
    pair_count = int(len(df))
    return {
        "method": method,
        "stage": stage,
        "protocol": protocol,
        "pair_count": pair_count,
        "genuine_count": int((labels == 1).sum()),
        "impostor_count": int((labels == 0).sum()),
        "scored_count": int(np.isfinite(scores).sum()),
        "failure_count": int((~np.isfinite(scores)).sum()),
        "failed_count": int((~np.isfinite(scores)).sum()),
        "roc_auc": auc,
        "eer": eer,
        "genuine_score_mean": gen_stats["mean"],
        "genuine_score_std": gen_stats["std"],
        "genuine_score_median": gen_stats["median"],
        "impostor_score_mean": imp_stats["mean"],
        "impostor_score_std": imp_stats["std"],
        "impostor_score_median": imp_stats["median"],
        "elapsed_seconds": float(elapsed_seconds),
        "runtime_ms_per_pair": float(1000.0 * elapsed_seconds / pair_count) if pair_count else float("nan"),
    }


def evaluate_matchers(
    *,
    stage: str,
    pair_bundle: dict[str, pd.DataFrame],
    tensors: TensorDescriptorCache,
    device: torch.device,
    score_batch_size: int,
    l3_shift_radius: int,
    methods: Iterable[str] = ALL_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    clcb_details: list[pd.DataFrame] = []
    for method in methods:
        for protocol in CONTROL_PROTOCOLS:
            df = pair_bundle[protocol].reset_index(drop=True).copy()
            if "protocol_id" not in df.columns:
                df["protocol_id"] = protocol
            scores, details, elapsed = score_pair_frame(
                method=method,
                df=df,
                tensors=tensors,
                device=device,
                batch_size=score_batch_size,
                l3_shift_radius=l3_shift_radius,
            )
            metric_rows.append(
                metric_row_from_scores(
                    method=method,
                    stage=stage,
                    protocol=protocol,
                    df=df,
                    scores=scores,
                    elapsed_seconds=elapsed,
                )
            )
            runtime_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "protocol": protocol,
                    "pair_count": int(len(df)),
                    "elapsed_seconds": float(elapsed),
                    "runtime_ms_per_pair": float(1000.0 * elapsed / len(df)) if len(df) else float("nan"),
                }
            )
            if protocol == "contactless_to_contact_based":
                clcb_details.append(details)
    detail_df = pd.concat(clcb_details, ignore_index=True) if clcb_details else pd.DataFrame()
    diagnostics = aggregate_local_diagnostics(detail_df, stage=stage)
    return pd.DataFrame(metric_rows), diagnostics, pd.DataFrame(runtime_rows)


def aggregate_local_diagnostics(detail_df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, method_df in detail_df.groupby("method", sort=False):
        for label, group_name in ((1, "genuine"), (0, "impostor")):
            group = method_df[method_df["label"].astype(int) == label].copy()
            if group.empty:
                continue
            matched_mean = pd.to_numeric(group["mean_matched_cosine"], errors="coerce").to_numpy(dtype=float)
            matched_median = pd.to_numeric(group["median_matched_cosine"], errors="coerce").to_numpy(dtype=float)
            mutual_count = pd.to_numeric(group["mutual_match_count"], errors="coerce").to_numpy(dtype=float)
            best_dx = pd.to_numeric(group["best_dx"], errors="coerce").to_numpy(dtype=float)
            best_dy = pd.to_numeric(group["best_dy"], errors="coerce").to_numpy(dtype=float)
            aligned = pd.to_numeric(group["best_aligned_cosine"], errors="coerce").to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "protocol": "contactless_to_contact_based",
                    "pair_group": group_name,
                    "pair_count": int(len(group)),
                    "no_valid_local_match_fraction": float(np.mean(~np.isfinite(matched_mean)))
                    if method == L2_MNN
                    else float(np.mean(pd.to_numeric(group["failed"], errors="coerce").fillna(1).astype(bool))),
                    "mutual_match_count_mean": _nanmean(mutual_count),
                    "mutual_match_count_std": _nanstd(mutual_count),
                    "mutual_match_count_median": _nanmedian(mutual_count),
                    "mean_matched_cosine_mean": _nanmean(matched_mean),
                    "mean_matched_cosine_median": _nanmedian(matched_mean),
                    "median_matched_cosine_median": _nanmedian(matched_median),
                    "best_dx_mean": _nanmean(best_dx),
                    "best_dx_std": _nanstd(best_dx),
                    "best_dx_median": _nanmedian(best_dx),
                    "best_dy_mean": _nanmean(best_dy),
                    "best_dy_std": _nanstd(best_dy),
                    "best_dy_median": _nanmedian(best_dy),
                    "best_aligned_cosine_mean": _nanmean(aligned),
                    "best_aligned_cosine_median": _nanmedian(aligned),
                }
            )
    return pd.DataFrame(rows)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _nanmean(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else float("nan"))


def _nanmedian(values: np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _retrieval_from_matrix(
    *,
    method: str,
    stage: str,
    direction: str,
    sim: np.ndarray,
    identity_count: int,
) -> dict[str, Any]:
    ranks: list[int] = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i], kind="mergesort")
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=int)
    return {
        "method": method,
        "stage": stage,
        "direction": direction,
        "identity_count": int(identity_count),
        "recall_at_1": float(np.mean(ranks_np <= 1)),
        "recall_at_5": float(np.mean(ranks_np <= min(5, identity_count))),
        "mrr": float(np.mean(1.0 / ranks_np)),
    }


def retrieval_metrics_for_matchers(
    *,
    stage: str,
    table: pd.DataFrame,
    tensors: TensorDescriptorCache,
    device: torch.device,
    score_batch_size: int,
    l3_shift_radius: int,
    methods: Iterable[str] = ALL_METHODS,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    ids = table["finger_unit_id"].astype(str).tolist()
    cl = table["cl_uid"].astype(str).tolist()
    cb = table["cb_uid"].astype(str).tolist()
    n = len(ids)
    for method in methods:
        for probe, gallery, direction in (
            (cl, cb, "CL_probe_to_CB_gallery"),
            (cb, cl, "CB_probe_to_CL_gallery"),
        ):
            pair_rows = [
                {
                    "pair_id": f"{direction}|{i:04d}|{j:04d}",
                    "label": int(i == j),
                    "sample_uid_a": probe_uid,
                    "sample_uid_b": gallery_uid,
                    "finger_unit_a": ids[i],
                    "finger_unit_b": ids[j],
                    "protocol_id": direction,
                }
                for i, probe_uid in enumerate(probe)
                for j, gallery_uid in enumerate(gallery)
            ]
            pair_df = pd.DataFrame(pair_rows)
            scores, _details, _elapsed = score_pair_frame(
                method=method,
                df=pair_df,
                tensors=tensors,
                device=device,
                batch_size=score_batch_size,
                l3_shift_radius=l3_shift_radius,
            )
            rows.append(
                _retrieval_from_matrix(
                    method=method,
                    stage=stage,
                    direction=direction,
                    sim=scores.reshape(n, n),
                    identity_count=n,
                )
            )
    return pd.DataFrame(rows)


def classify_local_signal(
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    *,
    cfg: LocalAuditConfig,
) -> dict[str, Any]:
    inner_metrics = metrics[metrics["stage"] == "inner_dev"].copy()
    inner_retrieval = retrieval[retrieval["stage"] == "inner_dev"].copy()
    clcb = inner_metrics[inner_metrics["protocol"] == "contactless_to_contact_based"].copy()
    if clcb.empty or L0_GLOBAL not in set(clcb["method"]):
        raise LocalCorrespondenceError("Missing inner-dev CL->CB metrics for L0/local decision")

    def _clcb_auc(method: str) -> float:
        row = clcb[clcb["method"] == method]
        return float(row.iloc[0]["roc_auc"]) if not row.empty else float("nan")

    def _within_mean(method: str) -> float:
        rows = inner_metrics[(inner_metrics["method"] == method) & (inner_metrics["protocol"].isin(WITHIN_PROTOCOLS))]
        vals = pd.to_numeric(rows["roc_auc"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else float("nan")

    def _mean_mrr(method: str) -> float:
        rows = inner_retrieval[inner_retrieval["method"] == method]
        vals = pd.to_numeric(rows["mrr"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else float("nan")

    def _direction_mrrs(method: str) -> dict[str, float]:
        rows = inner_retrieval[inner_retrieval["method"] == method]
        return {str(r["direction"]): float(r["mrr"]) for r in rows.to_dict("records")}

    l0_auc = _clcb_auc(L0_GLOBAL)
    l0_mrr = _mean_mrr(L0_GLOBAL)
    l0_within = _within_mean(L0_GLOBAL)
    l0_direction_mrr = _direction_mrrs(L0_GLOBAL)
    details: list[dict[str, Any]] = []
    a_like: list[str] = []
    for method in LOCAL_METHODS:
        auc = _clcb_auc(method)
        mean_mrr = _mean_mrr(method)
        within = _within_mean(method)
        direction_mrr = _direction_mrrs(method)
        auc_gain = auc - l0_auc if math.isfinite(auc) and math.isfinite(l0_auc) else float("nan")
        mrr_gain = mean_mrr - l0_mrr if math.isfinite(mean_mrr) and math.isfinite(l0_mrr) else float("nan")
        direction_ok = True
        for direction, l0_value in l0_direction_mrr.items():
            value = direction_mrr.get(direction, float("nan"))
            if not (math.isfinite(value) and math.isfinite(l0_value) and value >= l0_value + cfg.retrieval_close_margin):
                direction_ok = False
        within_ok = (
            math.isfinite(within)
            and (not math.isfinite(l0_within) or within >= l0_within - cfg.within_auc_drop_tolerance)
        )
        reaches = (
            math.isfinite(auc_gain)
            and auc_gain >= cfg.material_auc_gain
            and math.isfinite(mrr_gain)
            and mrr_gain >= cfg.retrieval_mrr_gain
            and direction_ok
            and within_ok
        )
        if reaches:
            a_like.append(method)
        details.append(
            {
                "method": method,
                "inner_dev_clcb_auc": auc,
                "auc_minus_l0": auc_gain,
                "inner_dev_mean_cross_modal_mrr": mean_mrr,
                "mrr_minus_l0": mrr_gain,
                "directional_retrieval_consistent": bool(direction_ok),
                "within_mean_auc": within,
                "within_modality_ok": bool(within_ok),
                "meets_material_local_signal_gate": bool(reaches),
            }
        )

    non_geometry_reaches = [m for m in a_like if m != L3_SPATIAL]
    if non_geometry_reaches:
        label = "A. LOCAL_SIGNAL_PRESENT"
    elif L3_SPATIAL in a_like:
        label = "B. GEOMETRY_HELPS_PARTIALLY"
    else:
        local_auc_close = all(
            math.isfinite(d["auc_minus_l0"]) and abs(float(d["auc_minus_l0"])) <= cfg.close_auc_margin
            for d in details
        )
        local_mrr_close = all(
            (not math.isfinite(d["mrr_minus_l0"])) or abs(float(d["mrr_minus_l0"])) <= cfg.retrieval_close_margin
            for d in details
        )
        label = "C. LOCAL_SIGNAL_WEAK" if local_auc_close and local_mrr_close else "D. INCONCLUSIVE"

    val_gate_opened = label.startswith("A.") or label.startswith("B.")
    selected = ""
    selected_reason = "Official VAL remains closed because A/B was not reached."
    if val_gate_opened:
        best_local_auc = max(_clcb_auc(method) for method in LOCAL_METHODS)
        eligible = [
            method
            for method in LOCAL_METHODS
            if math.isfinite(_clcb_auc(method))
            and _clcb_auc(method) >= best_local_auc - cfg.local_selection_auc_margin
        ]
        selected = sorted(eligible, key=lambda method: METHOD_SIMPLICITY_RANK[method])[0]
        selected_reason = (
            "A/B reached: select the simplest local matcher within 0.02 inner-dev CL->CB AUC of the best local matcher."
        )

    return {
        "classification": label,
        "criteria": {
            "l0_inner_dev_clcb_auc": l0_auc,
            "l0_inner_dev_mean_cross_modal_mrr": l0_mrr,
            "l0_inner_dev_within_mean_auc": l0_within,
            "material_auc_gain": cfg.material_auc_gain,
            "retrieval_mrr_gain": cfg.retrieval_mrr_gain,
            "directional_retrieval_margin": cfg.retrieval_close_margin,
            "within_auc_drop_tolerance": cfg.within_auc_drop_tolerance,
            "local_close_auc_margin": cfg.close_auc_margin,
        },
        "method_details": details,
        "official_val_gate": {
            "opened": bool(val_gate_opened),
            "reason": selected_reason,
            "selected_matcher": selected,
            "selection_rule": (
                "If A/B is reached, choose the simplest local matcher within 0.02 inner-dev CL->CB AUC of the best "
                "local matcher; evaluate only that frozen configuration on official VAL once."
            ),
        },
    }


def load_image_store_for_rows(images: pd.DataFrame, *, input_size: int) -> dict[str, torch.Tensor]:
    unique = images.drop_duplicates("sample_uid").sort_values("sample_uid", kind="mergesort")
    store: dict[str, torch.Tensor] = {}
    start = time.perf_counter()
    for i, row in enumerate(unique.to_dict("records")):
        if i and i % 500 == 0:
            print(f"[images] loaded {i}/{len(unique)} elapsed={time.perf_counter() - start:.1f}s", flush=True)
        store[str(row["sample_uid"])] = load_image_u8(Path(row["resolved_path"]), int(input_size))
    print(f"[images] loaded {len(store)} images into uint8 memory cache", flush=True)
    return store


def _unique_uids_for_pairs(bundle: dict[str, pd.DataFrame], images: pd.DataFrame) -> list[str]:
    uids = set(images["sample_uid"].astype(str).tolist())
    for df in bundle.values():
        uids.update(df["sample_uid_a"].astype(str).tolist())
        uids.update(df["sample_uid_b"].astype(str).tolist())
    return sorted(uids)


def _experiment_config_payload(cfg: LocalAuditConfig, *, checkpoint: Path, phase4b1_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_required": DEFAULT_CHECKPOINT_SHA256,
        "phase4b1_inner_split_source": str(Path(phase4b1_dir) / "inner_split.json"),
        "preprocess_contract": base.PREPROCESS_CONTRACT,
        "selected_feature_stage": SELECTED_STAGE_NAME,
        "selected_stage_index": int(cfg.selected_stage_index),
        "matchers": {
            L0_GLOBAL: "L2-normalized frozen ConvEncoder global embedding cosine.",
            L1_CHAMFER: "0.5 * (mean_i max_j cos(a_i,b_j) + mean_j max_i cos(a_i,b_j)).",
            L2_MNN: "coverage * mean mutual-nearest-neighbor cosine, coverage=count/min(num_a,num_b); no matches score -1.",
            L3_SPATIAL: (
                "Max mean same-grid local cosine over fixed coarse translations "
                f"dy,dx in [-{cfg.l3_shift_radius},{cfg.l3_shift_radius}]."
            ),
        },
        "decision_thresholds": {
            "material_auc_gain": cfg.material_auc_gain,
            "close_auc_margin": cfg.close_auc_margin,
            "retrieval_mrr_gain": cfg.retrieval_mrr_gain,
            "retrieval_close_margin": cfg.retrieval_close_margin,
            "within_auc_drop_tolerance": cfg.within_auc_drop_tolerance,
            "local_selection_auc_margin": cfg.local_selection_auc_margin,
        },
        "training": {
            "performed": False,
            "optimizer_instantiated": False,
            "encoder_frozen": True,
        },
        "official_val_gate": "VAL is opened only if A. LOCAL_SIGNAL_PRESENT or B. GEOMETRY_HELPS_PARTIALLY is reached.",
        "test_policy": "TEST is never read.",
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
    cfg: LocalAuditConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_arg)
    checkpoint = Path(checkpoint)
    write_json(outdir / "experiment_config.json", _experiment_config_payload(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir))

    train_images_all, resolved_root = load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[TRAIN])
    train_ids = sorted(train_images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = load_fixed_inner_split(train_ids, phase4b1_dir, split_seed=cfg.seed)
    inner_dev_ids = inner_split["inner_dev"]
    inner_dev_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    if smoke:
        keep_ids = inner_dev_ids[: min(8, len(inner_dev_ids))]
        inner_dev_ids = keep_ids
        inner_dev_images = inner_dev_images[inner_dev_images["finger_unit_id"].astype(str).isin(set(keep_ids))].copy()

    pair_max_pos = min(cfg.eval_max_pos, 40) if smoke else cfg.eval_max_pos
    pair_neg_per_pos = min(cfg.eval_neg_per_pos, 1) if smoke else cfg.eval_neg_per_pos
    inner_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=pair_max_pos,
        neg_per_pos=pair_neg_per_pos,
        seed=cfg.seed,
    )
    inner_counts = base.validate_pair_bundle(inner_bundle, stage="inner_dev")
    retrieval_table = base.build_retrieval_table(inner_dev_images, inner_dev_ids)

    model, ckpt_args, checkpoint_meta, checkpoint_sha = load_frozen_pair_model(checkpoint=checkpoint, device=device)
    input_size = int(ckpt_args.get("input_size", 384))
    width = int(ckpt_args.get("width", 32))
    inventory = convencoder_feature_map_inventory(
        width=width,
        input_size=input_size,
        selected_stage_index=cfg.selected_stage_index,
    )
    write_csv(outdir / "encoder_feature_map_inventory.csv", inventory)

    image_store = load_image_store_for_rows(inner_dev_images, input_size=input_size)
    uids = _unique_uids_for_pairs(inner_bundle, inner_dev_images)
    descriptor_cache = extract_descriptor_cache(
        encoder=model.encoder,
        image_store=image_store,
        uids=uids,
        device=device,
        batch_size=cfg.eval_batch_size,
        selected_stage_index=cfg.selected_stage_index,
        amp=cfg.amp,
    )
    descriptor_outputs: dict[str, str] = {}
    if cfg.descriptor_disk_cache:
        descriptor_outputs = write_descriptor_cache_artifacts(outdir, descriptor_cache, inner_dev_images)
    tensors = descriptor_cache_to_tensors(descriptor_cache, device)

    metrics, diagnostics, runtime = evaluate_matchers(
        stage="inner_dev",
        pair_bundle=inner_bundle,
        tensors=tensors,
        device=device,
        score_batch_size=cfg.score_batch_size,
        l3_shift_radius=cfg.l3_shift_radius,
    )
    retrieval = retrieval_metrics_for_matchers(
        stage="inner_dev",
        table=retrieval_table,
        tensors=tensors,
        device=device,
        score_batch_size=cfg.score_batch_size,
        l3_shift_radius=cfg.l3_shift_radius,
    )
    decision = classify_local_signal(metrics, retrieval, cfg=cfg)

    write_csv(outdir / "local_match_metrics.csv", metrics)
    write_csv(outdir / "retrieval_metrics.csv", retrieval)
    write_csv(outdir / "local_match_diagnostics.csv", diagnostics)
    write_csv(outdir / "runtime_metrics.csv", runtime)
    write_json(outdir / "selection_decision.json", decision)

    official_val_outputs: dict[str, str] = {}
    official_val_counts: dict[str, Any] = {}
    if decision["official_val_gate"]["opened"]:
        selected = str(decision["official_val_gate"]["selected_matcher"])
        val_images, _val_root = load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[VAL])
        val_ids = sorted(val_images["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
        val_bundle = base.load_official_val_pair_bundle(manifest_dir, controls_dir)
        official_val_counts = base.validate_pair_bundle(val_bundle, stage="official_val")
        val_retrieval_table = base.build_retrieval_table(val_images, val_ids)
        val_image_store = load_image_store_for_rows(val_images, input_size=input_size)
        val_uids = _unique_uids_for_pairs(val_bundle, val_images)
        val_cache = extract_descriptor_cache(
            encoder=model.encoder,
            image_store=val_image_store,
            uids=val_uids,
            device=device,
            batch_size=cfg.eval_batch_size,
            selected_stage_index=cfg.selected_stage_index,
            amp=cfg.amp,
        )
        val_tensors = descriptor_cache_to_tensors(val_cache, device)
        val_metrics, val_diagnostics, val_runtime = evaluate_matchers(
            stage="official_val",
            pair_bundle=val_bundle,
            tensors=val_tensors,
            device=device,
            score_batch_size=cfg.score_batch_size,
            l3_shift_radius=cfg.l3_shift_radius,
            methods=[selected],
        )
        val_retrieval = retrieval_metrics_for_matchers(
            stage="official_val",
            table=val_retrieval_table,
            tensors=val_tensors,
            device=device,
            score_batch_size=cfg.score_batch_size,
            l3_shift_radius=cfg.l3_shift_radius,
            methods=[selected],
        )
        write_csv(outdir / "official_val_selected_metrics.csv", val_metrics)
        write_csv(outdir / "official_val_selected_retrieval_metrics.csv", val_retrieval)
        write_csv(outdir / "official_val_selected_local_match_diagnostics.csv", val_diagnostics)
        write_csv(outdir / "official_val_selected_runtime_metrics.csv", val_runtime)
        official_val_outputs = {
            "official_val_selected_metrics_csv": str(outdir / "official_val_selected_metrics.csv"),
            "official_val_selected_retrieval_metrics_csv": str(outdir / "official_val_selected_retrieval_metrics.csv"),
            "official_val_selected_local_match_diagnostics_csv": str(
                outdir / "official_val_selected_local_match_diagnostics.csv"
            ),
            "official_val_selected_runtime_metrics_csv": str(outdir / "official_val_selected_runtime_metrics.csv"),
        }

    canonical_files = {
        "manifest_csv": Path(manifest_dir) / "manifest.csv",
        "pairs_train_csv": Path(manifest_dir) / "pairs_train.csv",
        "checkpoint": checkpoint,
    }
    if decision["official_val_gate"]["opened"]:
        canonical_files["pairs_val_csv"] = Path(manifest_dir) / "pairs_val.csv"
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
        "preprocess_contract": base.PREPROCESS_CONTRACT,
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
        "selected_feature_map": {
            "stage_name": descriptor_cache.selected_stage_name,
            "height": int(descriptor_cache.height),
            "width": int(descriptor_cache.width),
            "channels": int(descriptor_cache.channels),
            "descriptor_count_per_image": int(descriptor_cache.height * descriptor_cache.width),
        },
        "pair_counts": {"inner_dev": inner_counts, "official_val": official_val_counts},
        "outputs": {
            "encoder_feature_map_inventory_csv": str(outdir / "encoder_feature_map_inventory.csv"),
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "local_match_metrics_csv": str(outdir / "local_match_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "local_match_diagnostics_csv": str(outdir / "local_match_diagnostics.csv"),
            "runtime_metrics_csv": str(outdir / "runtime_metrics.csv"),
            "selection_decision_json": str(outdir / "selection_decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
            **descriptor_outputs,
            **official_val_outputs,
        },
        "decision": decision,
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
        "canonical_artifacts_not_read": {
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "TEST must remain closed for Phase 4B.2A.",
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
            "official_val_gate_opened": bool(decision["official_val_gate"]["opened"]),
            "test_pairs_read": False,
            "test_images_loaded": False,
            "canonical_manifest_or_pairs_modified": False,
            "canonical_checkpoint_modified": False,
            "encoder_frozen": True,
            "optimizer_instantiated": False,
            "training_performed": False,
            "used_p2_primary": False,
            "used_fusion": False,
            "used_transformer_or_cross_attention": False,
            "used_learned_alignment": False,
            "broad_hyperparameter_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "metrics": metrics,
        "retrieval": retrieval,
        "diagnostics": diagnostics,
        "runtime": runtime,
        "decision": decision,
        "run_manifest": run_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.2A PolyU Cross local-correspondence feasibility audit.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=LocalAuditConfig.seed)
    p.add_argument("--eval_max_pos", type=int, default=LocalAuditConfig.eval_max_pos)
    p.add_argument("--eval_neg_per_pos", type=int, default=LocalAuditConfig.eval_neg_per_pos)
    p.add_argument("--eval_batch_size", type=int, default=LocalAuditConfig.eval_batch_size)
    p.add_argument("--score_batch_size", type=int, default=LocalAuditConfig.score_batch_size)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--skip_descriptor_disk_cache", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Fast protocol smoke with fewer inner-dev identities/pairs.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = LocalAuditConfig(
        seed=int(args.seed),
        eval_max_pos=int(args.eval_max_pos),
        eval_neg_per_pos=int(args.eval_neg_per_pos),
        eval_batch_size=int(args.eval_batch_size),
        score_batch_size=int(args.score_batch_size),
        amp=not bool(args.no_amp),
        descriptor_disk_cache=not bool(args.skip_descriptor_disk_cache),
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
    except (LocalCorrespondenceError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    metrics = result["metrics"]
    retrieval = result["retrieval"]
    print("\n=== PolyU Cross Phase 4B.2A local-correspondence feasibility audit complete ===")
    print(f"Output dir        : {result['outdir']}")
    print(f"Classification    : {decision['classification']}")
    print(f"Official VAL gate : {'opened' if decision['official_val_gate']['opened'] else 'closed'}")
    print("\nInner-dev CL->CB verification:")
    clcb = metrics[
        (metrics["stage"] == "inner_dev")
        & (metrics["protocol"] == "contactless_to_contact_based")
    ][["method", "roc_auc", "eer", "runtime_ms_per_pair"]]
    print(clcb.to_string(index=False))
    print("\nInner-dev retrieval:")
    print(retrieval[["method", "direction", "recall_at_1", "recall_at_5", "mrr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
