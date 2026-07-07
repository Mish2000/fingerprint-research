"""Phase 4B.2B weakly supervised learned local alignment for PolyU Cross.

This phase trains only lightweight projection heads on top of frozen SD300
ConvEncoder local features. It deliberately avoids transformers,
cross-attention, geometric unwarping, pixel-level patch correspondence labels,
Fusion/SourceAFIS targets, TEST, and official VAL model selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
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
from scripts.deep.score_fast_pair_ddp_splits import PairModel
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import PolyUCrossPairError


RUN_SCHEMA_VERSION = "polyu_cross_learned_local_alignment_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_learned_local_alignment_v0"
DEFAULT_CHECKPOINT_SHA256 = local.DEFAULT_CHECKPOINT_SHA256

CL = "CL"
CB = "CB"
A0 = "A0_frozen_baseline"
A1 = "A1_shared_projection"
A2 = "A2_modality_specific_projections"
A3 = "A3_partial_local_adaptation"
A1_RANDOM = "control_A1_random_projection_untrained"
A2_RANDOM = "control_A2_random_projection_untrained"
SHUFFLED_A2 = "control_shuffled_identity_A2"
TRAINABLE_CONDITIONS = (A1, A2)
BASELINE_F_RAW_L2_AUC = 0.5119
STAGE_NAME = "encoder.net.3"
STAGE_INDEX = 3
LOCAL_DESCRIPTOR_DIM = 256
LOCAL_GRID = 24
LOCAL_DESCRIPTOR_COUNT = 576
PROJECTION_DIM = 128


class LearnedLocalAlignmentError(RuntimeError):
    """Raised for protocol or artifact failures in Phase 4B.2B."""


@dataclass(frozen=True)
class LearnedLocalConfig:
    seed: int = 13
    split_seed: int = 1341
    batch_identities: int = 8
    projection_dim: int = PROJECTION_DIM
    topk_fraction: float = 0.10
    temperature: float = 0.07
    max_epochs: int = 15
    patience: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 64
    pair_score_batch_size: int = 16
    amp: bool = False
    signal_auc_threshold: float = 0.60
    signal_auc_gain: float = 0.05
    retrieval_mrr_gain: float = 0.03
    within_auc_drop_tolerance: float = 0.05
    per_dim_std_min: float = 1e-4
    near_identical_max_fraction: float = 0.05
    stage3_seeds: tuple[int, ...] = (13, 29, 47)
    a3_target_auc: float = 0.70


@dataclass
class LearnedLocalResult:
    condition: str
    seed: int
    stage: str
    best_epoch: int
    best_auc: float
    best_state_dict: dict[str, torch.Tensor]
    curve_rows: list[dict[str, Any]]
    metric_rows: list[dict[str, Any]]
    retrieval_rows: list[dict[str, Any]]
    within_rows: list[dict[str, Any]]
    local_diag_rows: list[dict[str, Any]]
    collapse_rows: list[dict[str, Any]]
    trainable_param_count: int
    total_param_count: int
    trainable_names: list[str]
    gradient_check: dict[str, Any]
    encoder_sha256_before: str
    encoder_sha256_after: str


@dataclass
class ProjectedDescriptorCache:
    sample_uids: list[str]
    uid_to_index: dict[str, int]
    descriptors: torch.Tensor
    pre_norms: torch.Tensor
    descriptor_dim: int


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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def state_sha256(state_dict: dict[str, torch.Tensor], *, prefix: str = "encoder.") -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict):
        if prefix and not name.startswith(prefix):
            continue
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class LocalProjectionModel(nn.Module):
    """Frozen ConvEncoder stage-F local features plus lightweight heads."""

    def __init__(
        self,
        pair_model: PairModel,
        *,
        condition: str,
        projection_dim: int,
        train_encoder_stage3: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = pair_model.encoder
        self.condition = str(condition)
        self.projection_dim = int(projection_dim)
        self.train_encoder_stage3 = bool(train_encoder_stage3)
        self.shared_projection: nn.Module | None = None
        self.cl_projection: nn.Module | None = None
        self.cb_projection: nn.Module | None = None
        if self.condition == A0:
            self.output_dim = LOCAL_DESCRIPTOR_DIM
        elif self.condition in (A1, A3):
            self.shared_projection = nn.Linear(LOCAL_DESCRIPTOR_DIM, self.projection_dim)
            self.output_dim = self.projection_dim
        elif self.condition == A2:
            self.cl_projection = nn.Linear(LOCAL_DESCRIPTOR_DIM, self.projection_dim)
            self.cb_projection = nn.Linear(LOCAL_DESCRIPTOR_DIM, self.projection_dim)
            self.output_dim = self.projection_dim
        else:
            raise LearnedLocalAlignmentError(f"Unknown learned-local condition: {condition}")
        self.configure_trainability()

    def configure_trainability(self) -> list[str]:
        for param in self.parameters():
            param.requires_grad_(False)
        if self.condition == A1 and self.shared_projection is not None:
            for param in self.shared_projection.parameters():
                param.requires_grad_(True)
        elif self.condition == A2:
            assert self.cl_projection is not None and self.cb_projection is not None
            for param in self.cl_projection.parameters():
                param.requires_grad_(True)
            for param in self.cb_projection.parameters():
                param.requires_grad_(True)
        elif self.condition == A3 and self.shared_projection is not None:
            for param in self.shared_projection.parameters():
                param.requires_grad_(True)
            for param in self.encoder.net[3].parameters():  # type: ignore[attr-defined]
                param.requires_grad_(True)
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def frozen_stage_features(self, x: torch.Tensor) -> torch.Tensor:
        net = self.encoder.net  # type: ignore[attr-defined]
        if self.train_encoder_stage3:
            with torch.no_grad():
                y = net[0](x)
                y = net[1](y)
                y = net[2](y)
            y = net[3](y)
        else:
            with torch.no_grad():
                y = net[0](x)
                y = net[1](y)
                y = net[2](y)
                y = net[3](y)
        return y

    def local_descriptors(self, x: torch.Tensor, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.frozen_stage_features(x)
        local_grid = fmap.float().permute(0, 2, 3, 1).contiguous()
        flat = local_grid.reshape(local_grid.shape[0], local_grid.shape[1] * local_grid.shape[2], local_grid.shape[3])
        if self.condition == A0:
            projected = flat
        elif self.condition in (A1, A3):
            assert self.shared_projection is not None
            projected = self.shared_projection(flat)
        else:
            if str(modality) == CL:
                assert self.cl_projection is not None
                projected = self.cl_projection(flat)
            else:
                assert self.cb_projection is not None
                projected = self.cb_projection(flat)
        pre_norm = torch.linalg.vector_norm(projected.float(), dim=2)
        return F.normalize(projected.float(), p=2, dim=2), pre_norm


def build_projection_model(
    *,
    checkpoint: Path,
    condition: str,
    projection_dim: int,
    device: torch.device,
    seed: int,
    train_encoder_stage3: bool = False,
) -> tuple[LocalProjectionModel, dict[str, Any], dict[str, Any], str]:
    set_seed(int(seed))
    pair_model, ckpt_args, checkpoint_meta, checkpoint_sha = local.load_frozen_pair_model(
        checkpoint=checkpoint,
        device=device,
        require_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    model = LocalProjectionModel(
        pair_model,
        condition=condition,
        projection_dim=projection_dim,
        train_encoder_stage3=train_encoder_stage3,
    ).to(device)
    model.eval()
    return model, ckpt_args, checkpoint_meta, checkpoint_sha


def local_pair_scores_from_descriptors(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    topk_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return symmetric top-k scores plus directional scores for pair batches.

    ``a`` and ``b`` are B x L x D normalized local descriptor tensors. The
    score is differentiable through the selected top-k local responses.
    """
    if a.ndim != 3 or b.ndim != 3:
        raise LearnedLocalAlignmentError(f"Expected BxLxD descriptors, got {tuple(a.shape)} and {tuple(b.shape)}")
    sim = torch.bmm(a, b.transpose(1, 2))
    a_to_b = torch.max(sim, dim=2).values
    b_to_a = torch.max(sim, dim=1).values
    k_a = max(1, int(math.ceil(float(topk_fraction) * int(a_to_b.shape[1]))))
    k_b = max(1, int(math.ceil(float(topk_fraction) * int(b_to_a.shape[1]))))
    top_a = torch.topk(a_to_b, k=min(k_a, int(a_to_b.shape[1])), dim=1).values.mean(dim=1)
    top_b = torch.topk(b_to_a, k=min(k_b, int(b_to_a.shape[1])), dim=1).values.mean(dim=1)
    return 0.5 * (top_a + top_b), top_a, top_b


def pair_score_matrix(
    cl_desc: torch.Tensor,
    cb_desc: torch.Tensor,
    *,
    topk_fraction: float,
    pair_chunk: int = 16,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    n_cl = int(cl_desc.shape[0])
    n_cb = int(cb_desc.shape[0])
    for i in range(n_cl):
        parts: list[torch.Tensor] = []
        for start in range(0, n_cb, int(pair_chunk)):
            end = min(n_cb, start + int(pair_chunk))
            a = cl_desc[i : i + 1].expand(end - start, -1, -1)
            b = cb_desc[start:end]
            score, _ab, _ba = local_pair_scores_from_descriptors(a, b, topk_fraction=topk_fraction)
            parts.append(score)
        rows.append(torch.cat(parts, dim=0))
    return torch.stack(rows, dim=0)


def symmetric_contrastive_loss(scores: torch.Tensor, *, temperature: float) -> torch.Tensor:
    target = torch.arange(scores.shape[0], device=scores.device)
    logits = scores / float(temperature)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def score_pair_frame_topk(
    *,
    df: pd.DataFrame,
    cache: ProjectedDescriptorCache,
    device: torch.device,
    batch_size: int,
    topk_fraction: float,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    scores = np.full(len(df), np.nan, dtype=np.float64)
    details: list[dict[str, Any]] = []
    idx_a, idx_b, valid = local._pair_indices(df, cache.uid_to_index)
    valid_positions = np.flatnonzero(valid)
    start_time = time.perf_counter()
    descriptors = cache.descriptors
    for start in range(0, len(valid_positions), int(batch_size)):
        pos = valid_positions[start : start + int(batch_size)]
        a_idx = torch.as_tensor(idx_a[pos], dtype=torch.long, device=device)
        b_idx = torch.as_tensor(idx_b[pos], dtype=torch.long, device=device)
        a = descriptors.index_select(0, a_idx)
        b = descriptors.index_select(0, b_idx)
        score_t, ab_t, ba_t = local_pair_scores_from_descriptors(a, b, topk_fraction=topk_fraction)
        score_np = score_t.detach().cpu().numpy()
        ab_np = ab_t.detach().cpu().numpy()
        ba_np = ba_t.detach().cpu().numpy()
        scores[pos] = score_np
        for row_pos, score, ab, ba in zip(pos, score_np, ab_np, ba_np):
            row = df.iloc[int(row_pos)]
            details.append(
                {
                    "_row_order": int(row_pos),
                    "pair_id": str(row.get("pair_id", "")),
                    "label": int(row.get("label", -1)),
                    "sample_uid_a": str(row.get("sample_uid_a", "")),
                    "sample_uid_b": str(row.get("sample_uid_b", "")),
                    "score": float(score),
                    "topk_a_to_b": float(ab),
                    "topk_b_to_a": float(ba),
                    "topk_symmetric": float(score),
                    "failed": False,
                }
            )
    for row_pos in np.flatnonzero(~valid):
        row = df.iloc[int(row_pos)]
        details.append(
            {
                "_row_order": int(row_pos),
                "pair_id": str(row.get("pair_id", "")),
                "label": int(row.get("label", -1)),
                "sample_uid_a": str(row.get("sample_uid_a", "")),
                "sample_uid_b": str(row.get("sample_uid_b", "")),
                "score": float("nan"),
                "topk_a_to_b": float("nan"),
                "topk_b_to_a": float("nan"),
                "topk_symmetric": float("nan"),
                "failed": True,
            }
        )
    elapsed = time.perf_counter() - start_time
    detail_df = pd.DataFrame(details)
    if not detail_df.empty:
        detail_df = detail_df.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])
    return scores, detail_df, float(elapsed)


@torch.inference_mode()
def extract_projected_cache(
    *,
    model: LocalProjectionModel,
    image_store: dict[str, torch.Tensor],
    images: pd.DataFrame,
    uids: list[str],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> ProjectedDescriptorCache:
    model.eval()
    modality_by_uid = {
        str(row["sample_uid"]): (CL if str(row["modality"]) == base.CONTACTLESS else CB)
        for row in images.drop_duplicates("sample_uid").to_dict("records")
    }
    descriptors: list[torch.Tensor] = []
    pre_norms: list[torch.Tensor] = []
    ordered: list[str] = []
    for modality in (CL, CB):
        mod_uids = [uid for uid in uids if modality_by_uid.get(str(uid)) == modality]
        for start in range(0, len(mod_uids), int(batch_size)):
            chunk = mod_uids[start : start + int(batch_size)]
            x = base.stack_batch(image_store, chunk, device=device)
            with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
                desc, norms = model.local_descriptors(x, modality)
            descriptors.append(desc.detach().float())
            pre_norms.append(norms.detach().float())
            ordered.extend([str(uid) for uid in chunk])
    if not descriptors:
        raise LearnedLocalAlignmentError("No descriptors extracted")
    desc_t = torch.cat(descriptors, dim=0)
    norm_t = torch.cat(pre_norms, dim=0)
    order = np.argsort(np.asarray(ordered, dtype=str), kind="mergesort")
    desc_t = desc_t[torch.as_tensor(order, dtype=torch.long, device=device)]
    norm_t = norm_t[torch.as_tensor(order, dtype=torch.long, device=device)]
    ordered_sorted = [ordered[int(i)] for i in order]
    return ProjectedDescriptorCache(
        sample_uids=ordered_sorted,
        uid_to_index={uid: i for i, uid in enumerate(ordered_sorted)},
        descriptors=desc_t,
        pre_norms=norm_t,
        descriptor_dim=int(desc_t.shape[2]),
    )


def collapse_diagnostics(
    *,
    condition: str,
    stage: str,
    seed: int,
    cache: ProjectedDescriptorCache,
) -> dict[str, Any]:
    desc = cache.descriptors.detach().float()
    norms = cache.pre_norms.detach().float()
    flat = desc.reshape(-1, desc.shape[2])
    dim_std = torch.std(flat, dim=0, unbiased=True) if flat.shape[0] > 1 else torch.zeros(flat.shape[1], device=flat.device)
    image_means = F.normalize(desc.mean(dim=1), p=2, dim=1)
    if image_means.shape[0] > 1:
        sim = image_means @ image_means.T
        tri = sim[torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device).unbind()]
        mean_inter = float(torch.mean(tri).detach().cpu())
    else:
        mean_inter = float("nan")
    sample_count = min(4096, int(flat.shape[0]))
    if sample_count > 1:
        indices = torch.linspace(0, flat.shape[0] - 1, sample_count, device=flat.device).long()
        sample = flat.index_select(0, indices)
        sim = sample @ sample.T
        tri = sim[torch.triu_indices(sample_count, sample_count, offset=1, device=sim.device).unbind()]
        near = float(torch.mean((tri > 0.9999).float()).detach().cpu())
    else:
        near = 0.0
    return {
        "condition": condition,
        "stage": stage,
        "seed": int(seed),
        "descriptor_count": int(flat.shape[0]),
        "descriptor_dim": int(flat.shape[1]),
        "per_dim_std_mean": float(torch.mean(dim_std).detach().cpu()),
        "per_dim_std_min": float(torch.min(dim_std).detach().cpu()),
        "per_dim_std_max": float(torch.max(dim_std).detach().cpu()),
        "mean_inter_image_descriptor_cosine": mean_inter,
        "near_identical_descriptor_fraction_sampled": near,
        "pre_norm_mean": float(torch.mean(norms).detach().cpu()),
        "pre_norm_std": float(torch.std(norms, unbiased=True).detach().cpu()) if norms.numel() > 1 else 0.0,
        "pre_norm_min": float(torch.min(norms).detach().cpu()),
        "pre_norm_max": float(torch.max(norms).detach().cpu()),
    }


def local_diagnostics_from_details(
    *,
    condition: str,
    stage: str,
    seed: int,
    detail_df: pd.DataFrame,
    score_matrix: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, group_name in ((1, "genuine"), (0, "impostor")):
        group = detail_df[detail_df["label"].astype(int) == label]
        if group.empty:
            continue
        values = pd.to_numeric(group["topk_symmetric"], errors="coerce").to_numpy(dtype=float)
        ab = pd.to_numeric(group["topk_a_to_b"], errors="coerce").to_numpy(dtype=float)
        ba = pd.to_numeric(group["topk_b_to_a"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition,
                "stage": stage,
                "seed": int(seed),
                "diagnostic_group": group_name,
                "count": int(len(group)),
                "topk_symmetric_mean": _nanmean(values),
                "topk_symmetric_std": _nanstd(values),
                "topk_symmetric_median": _nanmedian(values),
                "topk_a_to_b_mean": _nanmean(ab),
                "topk_b_to_a_mean": _nanmean(ba),
                "failed_fraction": float(pd.to_numeric(group["failed"], errors="coerce").fillna(1).astype(bool).mean()),
            }
        )
    if score_matrix is not None and score_matrix.size:
        diag = np.diag(score_matrix)
        off = score_matrix[~np.eye(score_matrix.shape[0], dtype=bool)]
        for name, values in (("score_matrix_diagonal", diag), ("score_matrix_off_diagonal", off)):
            rows.append(
                {
                    "condition": condition,
                    "stage": stage,
                    "seed": int(seed),
                    "diagnostic_group": name,
                    "count": int(values.size),
                    "topk_symmetric_mean": _nanmean(values),
                    "topk_symmetric_std": _nanstd(values),
                    "topk_symmetric_median": _nanmedian(values),
                    "topk_a_to_b_mean": float("nan"),
                    "topk_b_to_a_mean": float("nan"),
                    "failed_fraction": 0.0,
                }
            )
    return rows


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


def score_retrieval(
    *,
    condition: str,
    stage: str,
    seed: int,
    table: pd.DataFrame,
    cache: ProjectedDescriptorCache,
    device: torch.device,
    batch_size: int,
    topk_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    ids = table["finger_unit_id"].astype(str).tolist()
    cl = table["cl_uid"].astype(str).tolist()
    cb = table["cb_uid"].astype(str).tolist()
    n = len(ids)
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
            }
            for i, probe_uid in enumerate(probe)
            for j, gallery_uid in enumerate(gallery)
        ]
        scores, _details, _elapsed = score_pair_frame_topk(
            df=pd.DataFrame(pair_rows),
            cache=cache,
            device=device,
            batch_size=batch_size,
            topk_fraction=topk_fraction,
        )
        sim = scores.reshape(n, n)
        matrices[direction] = sim
        ranks: list[int] = []
        for i in range(n):
            order = np.argsort(-sim[i], kind="mergesort")
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)
        ranks_np = np.asarray(ranks, dtype=int)
        rows.append(
            {
                "condition": condition,
                "stage": stage,
                "seed": int(seed),
                "direction": direction,
                "identity_count": int(n),
                "recall_at_1": float(np.mean(ranks_np <= 1)),
                "recall_at_5": float(np.mean(ranks_np <= min(5, n))),
                "mrr": float(np.mean(1.0 / ranks_np)),
            }
        )
    return rows, matrices


def evaluate_model(
    *,
    model: LocalProjectionModel,
    condition: str,
    stage: str,
    seed: int,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    eval_images: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    device: torch.device,
    cfg: LearnedLocalConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uids = local._unique_uids_for_pairs(pair_bundle, eval_images)
    cache = extract_projected_cache(
        model=model,
        image_store=image_store,
        images=eval_images,
        uids=uids,
        device=device,
        batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
    )
    metric_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    clcb_detail: pd.DataFrame | None = None
    for protocol in base.CONTROL_PROTOCOLS:
        df = pair_bundle[protocol].reset_index(drop=True).copy()
        df["protocol_id"] = protocol
        scores, details, elapsed = score_pair_frame_topk(
            df=df,
            cache=cache,
            device=device,
            batch_size=cfg.pair_score_batch_size,
            topk_fraction=cfg.topk_fraction,
        )
        row = local.metric_row_from_scores(
            method="topk_symmetric_local_score",
            stage=stage,
            protocol=protocol,
            df=df,
            scores=scores,
            elapsed_seconds=elapsed,
        )
        row.update({"condition": condition, "seed": int(seed), "topk_fraction": cfg.topk_fraction})
        if protocol == "contactless_to_contact_based":
            metric_rows.append(row)
            clcb_detail = details
        else:
            within_rows.append(row)
    retrieval_rows, matrices = score_retrieval(
        condition=condition,
        stage=stage,
        seed=seed,
        table=retrieval_table,
        cache=cache,
        device=device,
        batch_size=cfg.pair_score_batch_size,
        topk_fraction=cfg.topk_fraction,
    )
    matrix = matrices.get("CL_probe_to_CB_gallery")
    if clcb_detail is not None:
        local_rows.extend(
            local_diagnostics_from_details(
                condition=condition,
                stage=stage,
                seed=seed,
                detail_df=clcb_detail,
                score_matrix=matrix,
            )
        )
    collapse_rows = [collapse_diagnostics(condition=condition, stage=stage, seed=seed, cache=cache)]
    return metric_rows, retrieval_rows, within_rows, local_rows, collapse_rows


def gradient_status(model: LocalProjectionModel) -> dict[str, Any]:
    trainable = []
    frozen_with_grad = []
    trainable_without_grad = []
    for name, param in model.named_parameters():
        grad_norm = float(param.grad.detach().float().norm().cpu()) if param.grad is not None else 0.0
        grad_finite = math.isfinite(grad_norm)
        if param.requires_grad:
            trainable.append({"name": name, "grad_norm": grad_norm, "finite": bool(grad_finite)})
            if param.grad is None or not grad_finite or grad_norm <= 0.0:
                trainable_without_grad.append(name)
        elif param.grad is not None and (not grad_finite or grad_norm > 0.0):
            frozen_with_grad.append({"name": name, "grad_norm": grad_norm, "finite": bool(grad_finite)})
    return {
        "trainable_parameter_gradients": trainable,
        "trainable_without_gradient": trainable_without_grad,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "gradients_only_in_intended_components": not frozen_with_grad and not trainable_without_grad,
    }


def train_condition(
    *,
    condition: str,
    seed: int,
    stage: str,
    checkpoint: Path,
    train_images: pd.DataFrame,
    train_ids: list[str],
    inner_dev_images: pd.DataFrame,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    device: torch.device,
    cfg: LearnedLocalConfig,
    max_epochs: int,
    patience: int,
    shuffled_identity: bool = False,
) -> LearnedLocalResult:
    model, _ckpt_args, _checkpoint_meta, _checkpoint_sha = build_projection_model(
        checkpoint=checkpoint,
        condition=condition,
        projection_dim=cfg.projection_dim,
        device=device,
        seed=seed,
    )
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise LearnedLocalAlignmentError(f"{condition} has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")
    pools = base.build_identity_pools(train_images, train_ids)
    encoder_before = state_sha256(model.state_dict(), prefix="encoder.")
    best_auc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: list[dict[str, Any]] = []
    best_retrieval: list[dict[str, Any]] = []
    best_within: list[dict[str, Any]] = []
    best_local_diag: list[dict[str, Any]] = []
    best_collapse: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    no_improve = 0
    gradient_check: dict[str, Any] = {}
    start_time = time.perf_counter()
    stage_condition_name = condition if not shuffled_identity else SHUFFLED_A2
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        model.encoder.eval()
        losses: list[float] = []
        for batch in base.epoch_batches(
            pools,
            train_ids,
            batch_identities=cfg.batch_identities,
            epoch=epoch,
            seed=seed,
            shuffled_identity=shuffled_identity,
        ):
            cl = base.stack_batch(image_store, batch["cl_uids"], device=device)
            cb = base.stack_batch(image_store, batch["cb_uids"], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                z_cl, _norm_cl = model.local_descriptors(cl, CL)
                z_cb, _norm_cb = model.local_descriptors(cb, CB)
                scores = pair_score_matrix(
                    z_cl,
                    z_cb,
                    topk_fraction=cfg.topk_fraction,
                    pair_chunk=cfg.pair_score_batch_size,
                )
                loss = symmetric_contrastive_loss(scores, temperature=cfg.temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_check = gradient_status(model)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        metric_rows, retrieval_rows, within_rows, local_rows, collapse_rows = evaluate_model(
            model=model,
            condition=stage_condition_name,
            stage=stage,
            seed=seed,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            eval_images=inner_dev_images,
            image_store=image_store,
            device=device,
            cfg=cfg,
        )
        auc = float(metric_rows[0]["roc_auc"]) if metric_rows else float("nan")
        curve_rows.append(
            {
                "condition": stage_condition_name,
                "stage": stage,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "inner_dev_clcb_auc": auc,
                "seconds_elapsed": float(time.perf_counter() - start_time),
                "shuffled_identity": bool(shuffled_identity),
            }
        )
        print(json.dumps(curve_rows[-1], ensure_ascii=True), flush=True)
        if math.isfinite(auc) and auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = copy.deepcopy(metric_rows)
            best_retrieval = copy.deepcopy(retrieval_rows)
            best_within = copy.deepcopy(within_rows)
            best_local_diag = copy.deepcopy(local_rows)
            best_collapse = copy.deepcopy(collapse_rows)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(patience):
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best_state is None:
        raise LearnedLocalAlignmentError(f"{condition} did not produce a finite inner-dev AUC")
    model.load_state_dict(best_state, strict=True)
    encoder_after = state_sha256(model.state_dict(), prefix="encoder.")
    return LearnedLocalResult(
        condition=stage_condition_name,
        seed=int(seed),
        stage=stage,
        best_epoch=best_epoch,
        best_auc=best_auc,
        best_state_dict=best_state,
        curve_rows=curve_rows,
        metric_rows=best_metrics,
        retrieval_rows=best_retrieval,
        within_rows=best_within,
        local_diag_rows=best_local_diag,
        collapse_rows=best_collapse,
        trainable_param_count=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        total_param_count=int(sum(p.numel() for p in model.parameters())),
        trainable_names=trainable_names,
        gradient_check=gradient_check,
        encoder_sha256_before=encoder_before,
        encoder_sha256_after=encoder_after,
    )


def evaluate_untrained_condition(
    *,
    condition: str,
    output_condition: str,
    seed: int,
    stage: str,
    checkpoint: Path,
    inner_dev_images: pd.DataFrame,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    device: torch.device,
    cfg: LearnedLocalConfig,
) -> LearnedLocalResult:
    model, _ckpt_args, _checkpoint_meta, _checkpoint_sha = build_projection_model(
        checkpoint=checkpoint,
        condition=condition,
        projection_dim=cfg.projection_dim,
        device=device,
        seed=seed,
    )
    encoder_sha = state_sha256(model.state_dict(), prefix="encoder.")
    metric_rows, retrieval_rows, within_rows, local_rows, collapse_rows = evaluate_model(
        model=model,
        condition=output_condition,
        stage=stage,
        seed=seed,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        eval_images=inner_dev_images,
        image_store=image_store,
        device=device,
        cfg=cfg,
    )
    return LearnedLocalResult(
        condition=output_condition,
        seed=seed,
        stage=stage,
        best_epoch=0,
        best_auc=float(metric_rows[0]["roc_auc"]),
        best_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        curve_rows=[],
        metric_rows=metric_rows,
        retrieval_rows=retrieval_rows,
        within_rows=within_rows,
        local_diag_rows=local_rows,
        collapse_rows=collapse_rows,
        trainable_param_count=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        total_param_count=int(sum(p.numel() for p in model.parameters())),
        trainable_names=[name for name, param in model.named_parameters() if param.requires_grad],
        gradient_check={"not_applicable": "untrained evaluation"},
        encoder_sha256_before=encoder_sha,
        encoder_sha256_after=encoder_sha,
    )


def retrieval_by_direction(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(row["direction"]): {
            "recall_at_1": float(row["recall_at_1"]),
            "recall_at_5": float(row["recall_at_5"]),
            "mrr": float(row["mrr"]),
        }
        for row in rows
    }


def mean_within_auc(rows: list[dict[str, Any]]) -> float:
    vals = [float(row["roc_auc"]) for row in rows if math.isfinite(float(row["roc_auc"]))]
    return float(np.mean(vals)) if vals else float("nan")


def signal_gate_for_result(result: LearnedLocalResult, baseline: LearnedLocalResult, cfg: LearnedLocalConfig) -> dict[str, Any]:
    auc = float(result.metric_rows[0]["roc_auc"]) if result.metric_rows else float("nan")
    baseline_auc = BASELINE_F_RAW_L2_AUC
    retrieval = retrieval_by_direction(result.retrieval_rows)
    baseline_retrieval = retrieval_by_direction(baseline.retrieval_rows)
    retrieval_ok = True
    retrieval_details: dict[str, Any] = {}
    for direction, row in retrieval.items():
        base_row = baseline_retrieval.get(direction, {})
        mrr_gain = float(row.get("mrr", float("nan"))) - float(base_row.get("mrr", float("nan")))
        recall1_ok = float(row.get("recall_at_1", 0.0)) >= float(base_row.get("recall_at_1", 0.0))
        recall5_ok = float(row.get("recall_at_5", 0.0)) >= float(base_row.get("recall_at_5", 0.0))
        ok = math.isfinite(mrr_gain) and mrr_gain >= cfg.retrieval_mrr_gain and recall1_ok and recall5_ok
        retrieval_ok = retrieval_ok and ok
        retrieval_details[direction] = {"mrr_gain": mrr_gain, "recall1_ok": recall1_ok, "recall5_ok": recall5_ok, "ok": ok}
    collapse = result.collapse_rows[0] if result.collapse_rows else {}
    no_collapse = bool(
        float(collapse.get("per_dim_std_mean", 0.0)) >= cfg.per_dim_std_min
        and float(collapse.get("near_identical_descriptor_fraction_sampled", 1.0)) <= cfg.near_identical_max_fraction
    )
    within = mean_within_auc(result.within_rows)
    baseline_within = mean_within_auc(baseline.within_rows)
    within_ok = math.isfinite(within) and (not math.isfinite(baseline_within) or within >= baseline_within - cfg.within_auc_drop_tolerance)
    encoder_unchanged = result.encoder_sha256_before == result.encoder_sha256_after
    gradient_ok = bool(result.gradient_check.get("gradients_only_in_intended_components", False))
    passed = bool(
        math.isfinite(auc)
        and auc >= cfg.signal_auc_threshold
        and auc >= baseline_auc + cfg.signal_auc_gain
        and retrieval_ok
        and no_collapse
        and within_ok
        and encoder_unchanged
        and gradient_ok
    )
    return {
        "condition": result.condition,
        "seed": int(result.seed),
        "stage": result.stage,
        "passed": passed,
        "inner_dev_clcb_auc": auc,
        "baseline_f_raw_l2_auc": baseline_auc,
        "auc_threshold_ok": bool(math.isfinite(auc) and auc >= cfg.signal_auc_threshold),
        "auc_gain_ok": bool(math.isfinite(auc) and auc >= baseline_auc + cfg.signal_auc_gain),
        "retrieval_ok": bool(retrieval_ok),
        "retrieval_details": retrieval_details,
        "no_descriptor_collapse": bool(no_collapse),
        "within_modality_ok": bool(within_ok),
        "within_mean_auc": within,
        "baseline_within_mean_auc": baseline_within,
        "encoder_sha256_unchanged": bool(encoder_unchanged),
        "gradient_check_ok": bool(gradient_ok),
    }


def aggregate_seed_results(results: list[LearnedLocalResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in sorted({r.condition for r in results}):
        vals = np.asarray([r.best_auc for r in results if r.condition == condition], dtype=float)
        rows.append(
            {
                "condition": condition,
                "seed_count": int(vals.size),
                "auc_mean": float(np.mean(vals)) if vals.size else float("nan"),
                "auc_median": float(np.median(vals)) if vals.size else float("nan"),
                "auc_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "auc_min": float(np.min(vals)) if vals.size else float("nan"),
                "auc_max": float(np.max(vals)) if vals.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def classify_overall(
    *,
    gate_rows: list[dict[str, Any]],
    stage3_results: list[LearnedLocalResult],
    a3_activated: bool,
) -> dict[str, Any]:
    passed = [row for row in gate_rows if row.get("passed")]
    a1 = [row for row in passed if row["condition"] == A1]
    a2 = [row for row in passed if row["condition"] == A2]
    if a1 and not a2:
        label = "A. SHARED_LOCAL_ALIGNMENT_SUFFICIENT"
    elif a1 and a2:
        a1_auc = max(float(row["inner_dev_clcb_auc"]) for row in a1)
        a2_auc = max(float(row["inner_dev_clcb_auc"]) for row in a2)
        label = "B. MODALITY_SPECIFIC_LOCAL_ALIGNMENT_REQUIRED" if a2_auc >= a1_auc + 0.02 else "A. SHARED_LOCAL_ALIGNMENT_SUFFICIENT"
    elif a2:
        label = "B. MODALITY_SPECIFIC_LOCAL_ALIGNMENT_REQUIRED"
    elif a3_activated and stage3_results:
        label = "C. PARTIAL_FEATURE_ADAPTATION_REQUIRED"
    else:
        label = "D. LEARNED_LOCAL_ALIGNMENT_INSUFFICIENT"
    return {
        "classification": label,
        "reason": (
            "A1/A2 did not satisfy the predeclared signal gate."
            if label.startswith("D.")
            else "At least one learned local condition satisfied the predeclared signal gate."
        ),
    }


def condition_inventory_rows(results: list[LearnedLocalResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    descriptions = {
        A0: "Frozen encoder.net.3 local descriptors, no projection training.",
        A1_RANDOM: "Random shared 1x1 projection control before training.",
        A2_RANDOM: "Random modality-specific 1x1 projection control before training.",
        A1: "Frozen encoder with one shared trainable Linear(256,128) local projection.",
        A2: "Frozen encoder with separate CL/CB trainable Linear(256,128) local projections.",
        SHUFFLED_A2: "One-epoch shuffled-identity A2 diagnostic control.",
    }
    for result in results:
        if result.condition in seen:
            continue
        seen.add(result.condition)
        rows.append(
            {
                "condition": result.condition,
                "description": descriptions.get(result.condition, ""),
                "stage": result.stage,
                "seed": result.seed,
                "best_epoch": result.best_epoch,
                "trainable_param_count": result.trainable_param_count,
                "total_param_count": result.total_param_count,
                "trainable_parameter_names": ";".join(result.trainable_names),
                "encoder_sha256_unchanged": result.encoder_sha256_before == result.encoder_sha256_after,
            }
        )
    return pd.DataFrame(rows)


def _experiment_config(cfg: LearnedLocalConfig, *, checkpoint: Path, phase4b1_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_required": DEFAULT_CHECKPOINT_SHA256,
        "phase4b1_inner_split_source": str(Path(phase4b1_dir) / "inner_split.json"),
        "feature_map": {
            "stage": STAGE_NAME,
            "shape": [LOCAL_GRID, LOCAL_GRID, LOCAL_DESCRIPTOR_DIM],
            "local_descriptors_per_image": LOCAL_DESCRIPTOR_COUNT,
            "preprocessing": base.PREPROCESS_CONTRACT,
            "p2_used_primary": False,
        },
        "conditions": {
            A0: "Frozen local baseline, no training.",
            A1: "Shared Linear(256,128) projection head over frozen features.",
            A2: "Separate CL/CB Linear(256,128) projection heads over frozen features.",
            A3: "Not automatic; only encoder.net.3 unfreeze if gated.",
        },
        "local_pair_score": {
            "name": "symmetric_topk_local_max_cosine",
            "topk_fraction": cfg.topk_fraction,
            "train_eval_same_score_definition": True,
        },
        "loss": {"name": "symmetric_cross_entropy_over_full_batch_score_matrix", "temperature": cfg.temperature},
        "batching": {"batch_identities": cfg.batch_identities, "identity_balanced": True},
        "optimizer": {
            "family": "AdamW",
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "broad_hyperparameter_search": False,
        },
        "training_funnel": {
            "stage1_smoke": {"conditions": [A1, A2], "seed": 13, "epochs": 1},
            "stage2_signal": {"conditions": [A1, A2], "seed": 13, "max_epochs": cfg.max_epochs, "patience": cfg.patience},
            "stage3_multiseed_only_if_signal_gate_passes": list(cfg.stage3_seeds),
        },
        "official_val_gate": "Closed unless signal gate passes and stable gains appear in at least 2/3 seeds.",
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
    cfg: LearnedLocalConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    device = local.resolve_device(device_arg)
    write_json(outdir / "experiment_config.json", _experiment_config(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir))
    set_seed(cfg.seed)

    train_images_all, resolved_root = local.load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[base.TRAIN])
    train_ids_all = sorted(train_images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = local.load_fixed_inner_split(train_ids_all, phase4b1_dir, split_seed=cfg.split_seed)
    inner_train_ids = inner_split["inner_train"]
    inner_dev_ids = inner_split["inner_dev"]
    if smoke:
        inner_train_ids = inner_train_ids[: min(24, len(inner_train_ids))]
        inner_dev_ids = inner_dev_ids[: min(8, len(inner_dev_ids))]
    train_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_train_ids))].copy()
    inner_dev_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    pair_max_pos = min(cfg.eval_max_pos, 40) if smoke else cfg.eval_max_pos
    pair_neg_per_pos = min(cfg.eval_neg_per_pos, 1) if smoke else cfg.eval_neg_per_pos
    pair_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=pair_max_pos,
        neg_per_pos=pair_neg_per_pos,
        seed=cfg.split_seed,
    )
    pair_counts = base.validate_pair_bundle(pair_bundle, stage="inner_dev")
    retrieval_table = base.build_retrieval_table(inner_dev_images, inner_dev_ids)
    all_needed_images = pd.concat([train_images, inner_dev_images], axis=0).drop_duplicates("sample_uid")
    ckpt_payload, checkpoint_meta = base.load_checkpoint_payload(checkpoint)
    ckpt_args = dict(ckpt_payload.get("args", {}) or {})
    input_size = int(ckpt_args.get("input_size", 384))
    image_store = local.load_image_store_for_rows(all_needed_images, input_size=input_size)
    checkpoint_sha = sha256_file(checkpoint) or ""
    if checkpoint_sha != DEFAULT_CHECKPOINT_SHA256:
        raise LearnedLocalAlignmentError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {DEFAULT_CHECKPOINT_SHA256}")

    write_json(
        outdir / "inner_split.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "source": str(Path(phase4b1_dir) / "inner_split.json"),
            "inner_train": inner_train_ids,
            "inner_dev": inner_dev_ids,
            "identity_disjoint": not bool(set(inner_train_ids).intersection(inner_dev_ids)),
            "official_val_read": False,
            "test_used": False,
        },
    )

    results: list[LearnedLocalResult] = []
    print("[eval] A0 frozen baseline", flush=True)
    a0 = evaluate_untrained_condition(
        condition=A0,
        output_condition=A0,
        seed=cfg.seed,
        stage="baseline",
        checkpoint=checkpoint,
        inner_dev_images=inner_dev_images,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        image_store=image_store,
        device=device,
        cfg=cfg,
    )
    results.append(a0)

    for condition, output in ((A1, A1_RANDOM), (A2, A2_RANDOM)):
        print(f"[eval] {output}", flush=True)
        results.append(
            evaluate_untrained_condition(
                condition=condition,
                output_condition=output,
                seed=13,
                stage="random_projection_control",
                checkpoint=checkpoint,
                inner_dev_images=inner_dev_images,
                pair_bundle=pair_bundle,
                retrieval_table=retrieval_table,
                image_store=image_store,
                device=device,
                cfg=cfg,
            )
        )

    stage1_results: list[LearnedLocalResult] = []
    for condition in (A1, A2):
        print(f"[stage1] {condition}", flush=True)
        result = train_condition(
            condition=condition,
            seed=13,
            stage="stage1_smoke",
            checkpoint=checkpoint,
            train_images=train_images,
            train_ids=inner_train_ids,
            inner_dev_images=inner_dev_images,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            image_store=image_store,
            device=device,
            cfg=cfg,
            max_epochs=1,
            patience=1,
        )
        stage1_results.append(result)
        results.append(result)

    print("[control] shuffled identity A2", flush=True)
    shuffled = train_condition(
        condition=A2,
        seed=13013,
        stage="shuffled_identity_control",
        checkpoint=checkpoint,
        train_images=train_images,
        train_ids=inner_train_ids,
        inner_dev_images=inner_dev_images,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        image_store=image_store,
        device=device,
        cfg=cfg,
        max_epochs=1,
        patience=1,
        shuffled_identity=True,
    )
    results.append(shuffled)

    stage2_results: list[LearnedLocalResult] = []
    stage2_max_epochs = min(2, cfg.max_epochs) if smoke else cfg.max_epochs
    stage2_patience = 1 if smoke else cfg.patience
    for condition in (A1, A2):
        print(f"[stage2] {condition}", flush=True)
        result = train_condition(
            condition=condition,
            seed=13,
            stage="stage2_signal",
            checkpoint=checkpoint,
            train_images=train_images,
            train_ids=inner_train_ids,
            inner_dev_images=inner_dev_images,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            image_store=image_store,
            device=device,
            cfg=cfg,
            max_epochs=stage2_max_epochs,
            patience=stage2_patience,
        )
        stage2_results.append(result)
        results.append(result)

    gate_rows = [signal_gate_for_result(result, a0, cfg) for result in stage2_results]
    passing_conditions = [row["condition"] for row in gate_rows if row.get("passed")]
    stage3_results: list[LearnedLocalResult] = []
    stable_gate: dict[str, Any] = {"opened": False, "reason": "No Stage-2 condition passed the signal gate."}
    if passing_conditions and not smoke:
        for condition in sorted(set(passing_conditions)):
            for seed in cfg.stage3_seeds:
                existing = next((r for r in stage2_results if r.condition == condition and r.seed == seed), None)
                if existing is not None:
                    stage3_results.append(existing)
                    continue
                print(f"[stage3] {condition} seed={seed}", flush=True)
                stage3_results.append(
                    train_condition(
                        condition=condition,
                        seed=seed,
                        stage="stage3_multiseed",
                        checkpoint=checkpoint,
                        train_images=train_images,
                        train_ids=inner_train_ids,
                        inner_dev_images=inner_dev_images,
                        pair_bundle=pair_bundle,
                        retrieval_table=retrieval_table,
                        image_store=image_store,
                        device=device,
                        cfg=cfg,
                        max_epochs=cfg.max_epochs,
                        patience=cfg.patience,
                    )
                )
        results.extend([r for r in stage3_results if r not in results])
        stable_rows = []
        for result in stage3_results:
            stable_rows.append(signal_gate_for_result(result, a0, cfg))
        stable_by_condition: dict[str, int] = {}
        for row in stable_rows:
            stable_by_condition[row["condition"]] = stable_by_condition.get(row["condition"], 0) + int(bool(row["passed"]))
        stable_gate = {
            "opened": any(count >= 2 for count in stable_by_condition.values()),
            "stable_pass_counts": stable_by_condition,
            "reason": "Stable gains in at least 2/3 seeds." if any(count >= 2 for count in stable_by_condition.values()) else "No condition passed in at least 2/3 seeds.",
        }

    a3_activated = False
    a3_reason = "A2 did not pass the Stage-2 signal gate; A3 remains inactive."
    if any(row.get("passed") and row["condition"] == A2 for row in gate_rows):
        a2_stage3 = [r for r in stage3_results if r.condition == A2]
        median_auc = float(np.median([r.best_auc for r in a2_stage3])) if a2_stage3 else float(next(row["inner_dev_clcb_auc"] for row in gate_rows if row["condition"] == A2))
        if median_auc < cfg.a3_target_auc:
            a3_activated = True
            a3_reason = f"A2 passed but median AUC {median_auc:.4f} remains below target {cfg.a3_target_auc:.2f}; A3 would be activated in a dedicated follow-up run."
        else:
            a3_reason = f"A2 passed and median AUC {median_auc:.4f} reached target; A3 not activated."
    overall = classify_overall(gate_rows=gate_rows, stage3_results=stage3_results, a3_activated=False)
    decision = {
        **overall,
        "stage1_smoke": {
            "conditions": [
                {
                    "condition": r.condition,
                    "best_epoch": r.best_epoch,
                    "inner_dev_clcb_auc": r.best_auc,
                    "finite_loss": all(math.isfinite(float(row["train_loss"])) for row in r.curve_rows),
                    "encoder_sha256_unchanged": r.encoder_sha256_before == r.encoder_sha256_after,
                    "gradient_check": r.gradient_check,
                }
                for r in stage1_results
            ]
        },
        "stage2_signal_gate": gate_rows,
        "stage3_multiseed": {
            "ran": bool(stage3_results and not smoke),
            "passing_conditions_from_stage2": passing_conditions,
            "stable_gate": stable_gate,
            "aggregate": aggregate_seed_results(stage3_results).to_dict("records") if stage3_results else [],
        },
        "a3_gate": {
            "activated": bool(a3_activated and False),
            "activation_candidate": bool(a3_activated),
            "reason": a3_reason if a3_activated else a3_reason,
            "note": "A3 is not run automatically unless the predeclared A2 gate is satisfied; this run did not execute A3.",
        },
        "official_val_gate": {
            "opened": False,
            "reason": (
                "Official VAL remains closed because stable Stage-3 gains were not established."
                if not stable_gate.get("opened")
                else "Stable gains reached; official VAL final training/evaluation is intentionally left for the gated branch."
            ),
        },
    }

    metric_rows = [row for r in results for row in r.metric_rows]
    retrieval_rows = [row for r in results for row in r.retrieval_rows]
    within_rows = [row for r in results for row in r.within_rows]
    local_rows = [row for r in results for row in r.local_diag_rows]
    collapse_rows = [row for r in results for row in r.collapse_rows]
    curve_rows = [row for r in results for row in r.curve_rows]
    inventory = condition_inventory_rows(results)
    write_csv(outdir / "condition_inventory.csv", inventory)
    write_csv(outdir / "training_curves.csv", curve_rows)
    write_csv(outdir / "inner_dev_metrics.csv", metric_rows)
    write_csv(outdir / "retrieval_metrics.csv", retrieval_rows)
    write_csv(outdir / "within_modality_controls.csv", within_rows)
    write_csv(outdir / "local_diagnostics.csv", local_rows)
    write_csv(outdir / "collapse_diagnostics.csv", collapse_rows)
    write_json(outdir / "signal_gate_decision.json", decision)

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
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "inner_split": {
            "source": str(Path(phase4b1_dir) / "inner_split.json"),
            "inner_train_identity_count": len(inner_train_ids),
            "inner_dev_identity_count": len(inner_dev_ids),
            "identity_disjoint": not bool(set(inner_train_ids).intersection(inner_dev_ids)),
            "smoke": bool(smoke),
        },
        "pair_counts": {"inner_dev": pair_counts},
        "feature_map": {"stage": STAGE_NAME, "shape": [24, 24, 256]},
        "config": _experiment_config(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir),
        "decision": decision,
        "outputs": {
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "inner_split_json": str(outdir / "inner_split.json"),
            "condition_inventory_csv": str(outdir / "condition_inventory.csv"),
            "training_curves_csv": str(outdir / "training_curves.csv"),
            "inner_dev_metrics_csv": str(outdir / "inner_dev_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "within_modality_controls_csv": str(outdir / "within_modality_controls.csv"),
            "local_diagnostics_csv": str(outdir / "local_diagnostics.csv"),
            "collapse_diagnostics_csv": str(outdir / "collapse_diagnostics.csv"),
            "signal_gate_decision_json": str(outdir / "signal_gate_decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
        },
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
        "canonical_artifacts_not_read": {
            "pairs_val_csv": str(Path(manifest_dir) / "pairs_val.csv"),
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "Official VAL remains gated and TEST remains closed for Phase 4B.2B.",
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
            "test_pairs_read": False,
            "test_images_loaded": False,
            "official_val_read": False,
            "official_val_used_for_early_stopping_or_selection": False,
            "canonical_manifest_or_pairs_modified": False,
            "canonical_checkpoint_modified": False,
            "used_p2_primary": False,
            "used_fusion_or_sourceafis_targets": False,
            "used_dataset_session_modality_as_model_inputs": False,
            "used_pixel_coordinate_correspondence_labels": False,
            "used_transformer_or_cross_attention": False,
            "used_geometric_unwarping": False,
            "broad_hyperparameter_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "decision": decision,
        "metrics": pd.DataFrame(metric_rows),
        "retrieval": pd.DataFrame(retrieval_rows),
        "within": pd.DataFrame(within_rows),
        "collapse": pd.DataFrame(collapse_rows),
        "inventory": inventory,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.2B learned local descriptor alignment for PolyU Cross.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_identities", type=int, default=LearnedLocalConfig.batch_identities)
    p.add_argument("--max_epochs", type=int, default=LearnedLocalConfig.max_epochs)
    p.add_argument("--patience", type=int, default=LearnedLocalConfig.patience)
    p.add_argument("--eval_batch_size", type=int, default=LearnedLocalConfig.eval_batch_size)
    p.add_argument("--pair_score_batch_size", type=int, default=LearnedLocalConfig.pair_score_batch_size)
    p.add_argument("--amp", action="store_true", help="Enable CUDA autocast/GradScaler. Disabled by default for GTX 1070 stability.")
    p.add_argument("--no_amp", action="store_true", help="Deprecated compatibility flag; AMP is already off unless --amp is set.")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = LearnedLocalConfig(
        batch_identities=int(args.batch_identities),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        eval_batch_size=int(args.eval_batch_size),
        pair_score_batch_size=int(args.pair_score_batch_size),
        amp=bool(args.amp) and not bool(args.no_amp),
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
    except (LearnedLocalAlignmentError, local.LocalCorrespondenceError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n=== PolyU Cross Phase 4B.2B learned local alignment complete ===")
    print(f"Output dir     : {result['outdir']}")
    print(f"Classification : {result['decision']['classification']}")
    metrics = result["metrics"]
    show = metrics[metrics["stage"].isin(["baseline", "stage2_signal"])][
        ["condition", "stage", "seed", "roc_auc", "eer", "runtime_ms_per_pair"]
    ]
    print("\nInner-dev CL->CB:")
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
