"""Phase 4B.3A minimal pair-conditioned local correspondence learning.

This diagnostic tests whether a single compact pair-conditioning block can
recover PolyU Cross CL->CB local correspondence when the canonical SD300
ConvEncoder remains frozen. It deliberately avoids TEST, official VAL
development, P2, Fusion/SourceAFIS/SIFT targets, encoder fine-tuning, geometric
unwarping, and broad architecture search.
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
from scripts.diagnostics import run_polyu_cross_differentiable_matcher_validity as matcher
from scripts.diagnostics import run_polyu_cross_learned_local_alignment as learned
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import PolyUCrossPairError


RUN_SCHEMA_VERSION = "polyu_cross_pair_conditioned_correspondence_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_pair_conditioned_correspondence_v0"
DEFAULT_CHECKPOINT_SHA256 = local.DEFAULT_CHECKPOINT_SHA256

C0 = "C0_frozen_sinkhorn_reference"
C1 = "C1_projection_sinkhorn"
C2 = "C2_pair_conditioned_sinkhorn"
C2_RANDOM = "control_C2_random_initialization"
C2_SHUFFLED = "control_C2_shuffled_identity"
TRAINABLE_CONDITIONS = (C1, C2)

CLCB = matcher.CLCB
CLCL_SAME = matcher.CLCL_SAME
CLCL_CROSS = matcher.CLCL_CROSS
CBCB_SAME = matcher.CBCB_SAME
CBCB_CROSS = matcher.CBCB_CROSS
WITHIN_PROTOCOLS = base.WITHIN_PROTOCOLS

BASELINE_HARD_MNN_F_RAW_AUC = 0.5119
BASELINE_HARD_MNN_RETRIEVAL = {
    "CL_probe_to_CB_gallery": {"recall_at_1": 0.050, "recall_at_5": 0.150, "mrr": 0.13813710984629662},
    "CB_probe_to_CL_gallery": {"recall_at_1": 0.025, "recall_at_5": 0.125, "mrr": 0.11021991262657298},
}


class PairConditionedError(RuntimeError):
    """Raised for Phase 4B.3A protocol or artifact failures."""


@dataclass(frozen=True)
class PairConditionedConfig:
    seed: int = 13
    split_seed: int = 1341
    batch_identities: int = 8
    negatives_per_anchor: int = 3
    projection_dim: int = 128
    attention_heads: int = 4
    attention_blocks: int = 1
    train_temperature: float = 0.07
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 4
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 64
    train_pair_chunk_size: int = 4
    eval_pair_chunk_size: int = 4
    amp: bool = False
    sinkhorn_entropy_regularization: float = 0.07
    sinkhorn_iterations: int = 20
    sinkhorn_dustbin_score: float = 0.0
    sinkhorn_dustbin_mass: float = 1.0
    dustbin_dominated_real_mass_threshold: float = 0.25
    signal_auc_threshold: float = 0.62
    signal_auc_gain_over_hard_mnn: float = 0.08
    retrieval_mrr_gain: float = 0.0
    no_collapse_variance_min: float = 1e-5
    no_collapse_near_identical_max: float = 0.05
    within_drop_tolerance_vs_c0: float = 0.15
    within_min_auc_floor: float = 0.55
    material_c2_auc_gain_over_c1: float = 0.02
    stage3_seeds: tuple[int, ...] = (13, 29, 47)


@dataclass
class TrainRunResult:
    condition: str
    stage: str
    seed: int
    best_epoch: int
    best_auc: float
    best_state_dict: dict[str, torch.Tensor]
    curve_rows: list[dict[str, Any]]
    metric_rows: list[dict[str, Any]]
    retrieval_rows: list[dict[str, Any]]
    within_rows: list[dict[str, Any]]
    correspondence_rows: list[dict[str, Any]]
    attention_rows: list[dict[str, Any]]
    trainable_param_count: int
    total_param_count: int
    trainable_names: list[str]
    gradient_check: dict[str, Any]
    encoder_sha256_before: str
    encoder_sha256_after: str


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


def make_2d_sincos_position_encoding(height: int, width: int, dim: int) -> torch.Tensor:
    if int(dim) % 4 != 0:
        raise PairConditionedError(f"2D positional encoding dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, int(height), dtype=torch.float32),
        torch.linspace(0.0, 1.0, int(width), dtype=torch.float32),
        indexing="ij",
    )
    half = int(dim) // 4
    freq = torch.exp(torch.linspace(0.0, math.log(10000.0), half, dtype=torch.float32))
    yv = y.reshape(-1, 1) / freq.reshape(1, -1)
    xv = x.reshape(-1, 1) / freq.reshape(1, -1)
    return torch.cat([torch.sin(yv), torch.cos(yv), torch.sin(xv), torch.cos(xv)], dim=1).contiguous()


class PairConditioningBlock(nn.Module):
    """Exactly one self-attention plus bidirectional cross-attention block."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)

    def _entropy(self, weights: torch.Tensor) -> torch.Tensor:
        # weights: B x heads x target_tokens x source_tokens
        probs = weights.clamp_min(torch.finfo(weights.dtype).eps)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy / math.log(max(2, int(weights.shape[-1])))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        need_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        need_weights = bool(need_diagnostics)
        a_self, wa = self.self_attn(
            a,
            a,
            a,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        b_self, wb = self.self_attn(
            b,
            b,
            b,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        a = self.self_norm(a + a_self)
        b = self.self_norm(b + b_self)
        a_cross, wab = self.cross_attn(
            a,
            b,
            b,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        b_cross, wba = self.cross_attn(
            b,
            a,
            a,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        a = self.cross_norm(a + a_cross)
        b = self.cross_norm(b + b_cross)
        diagnostics: dict[str, torch.Tensor] = {}
        if need_diagnostics:
            assert wa is not None and wb is not None and wab is not None and wba is not None
            diagnostics["self_attention_entropy"] = 0.5 * (
                self._entropy(wa).mean(dim=(1, 2)) + self._entropy(wb).mean(dim=(1, 2))
            )
            diagnostics["cross_attention_entropy"] = 0.5 * (
                self._entropy(wab).mean(dim=(1, 2)) + self._entropy(wba).mean(dim=(1, 2))
            )
        return a, b, diagnostics


class LocalCorrespondenceModel(nn.Module):
    """C1/C2 descriptor model operating only on frozen local descriptors."""

    def __init__(
        self,
        *,
        condition: str,
        projection_dim: int,
        attention_heads: int,
        grid_height: int = 24,
        grid_width: int = 24,
    ) -> None:
        super().__init__()
        if condition not in (C1, C2):
            raise PairConditionedError(f"Unknown trainable condition: {condition}")
        self.condition = str(condition)
        self.projection = nn.Linear(256, int(projection_dim))
        self.block: PairConditioningBlock | None = None
        if self.condition == C2:
            self.block = PairConditioningBlock(int(projection_dim), int(attention_heads))
        pe = make_2d_sincos_position_encoding(grid_height, grid_width, int(projection_dim))
        self.register_buffer("position_encoding", pe, persistent=False)

    def forward_pair(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        need_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if a.ndim != 3 or b.ndim != 3:
            raise PairConditionedError(f"Expected BxLxD descriptors, got {tuple(a.shape)} and {tuple(b.shape)}")
        za = self.projection(a.float())
        zb = self.projection(b.float())
        pre_norm_a = torch.linalg.vector_norm(za.float(), dim=2)
        pre_norm_b = torch.linalg.vector_norm(zb.float(), dim=2)
        attn_diag: dict[str, torch.Tensor] = {}
        if self.condition == C2:
            if self.block is None:
                raise PairConditionedError("C2 missing pair-conditioning block")
            pos = self.position_encoding.to(device=za.device, dtype=za.dtype).unsqueeze(0)
            za, zb, attn_diag = self.block(za + pos, zb + pos, need_diagnostics=need_diagnostics)
        za = F.normalize(za.float(), p=2, dim=2)
        zb = F.normalize(zb.float(), p=2, dim=2)
        diag = {
            "pre_norm_a_mean": pre_norm_a.mean(dim=1),
            "pre_norm_b_mean": pre_norm_b.mean(dim=1),
            "pre_norm_a_std": pre_norm_a.std(dim=1, unbiased=False),
            "pre_norm_b_std": pre_norm_b.std(dim=1, unbiased=False),
            **attn_diag,
        }
        return za, zb, diag


def trainable_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def count_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    return trainable, total


def gradient_status(model: nn.Module) -> dict[str, Any]:
    trainable: list[dict[str, Any]] = []
    frozen_with_grad: list[dict[str, Any]] = []
    trainable_without_grad: list[str] = []
    for name, param in model.named_parameters():
        grad_norm = float(param.grad.detach().float().norm().cpu()) if param.grad is not None else 0.0
        finite = math.isfinite(grad_norm)
        if param.requires_grad:
            trainable.append({"name": name, "grad_norm": grad_norm, "finite": bool(finite)})
            if param.grad is None or not finite or grad_norm <= 0.0:
                trainable_without_grad.append(name)
        elif param.grad is not None and (not finite or grad_norm > 0.0):
            frozen_with_grad.append({"name": name, "grad_norm": grad_norm, "finite": bool(finite)})
    return {
        "trainable_parameter_gradients": trainable,
        "trainable_without_gradient": trainable_without_grad,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "gradients_only_in_intended_components": not frozen_with_grad and not trainable_without_grad,
    }


def _positions(device: torch.device, length: int = 576, width: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(int(length), device=device, dtype=torch.float32)
    y = torch.floor(idx / int(width))
    x = torch.remainder(idx, int(width))
    return y, x


def sinkhorn_scores_and_diagnostics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cfg: PairConditionedConfig,
    return_diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if a.ndim != 3 or b.ndim != 3:
        raise PairConditionedError(f"Expected BxLxD descriptors, got {tuple(a.shape)} and {tuple(b.shape)}")
    eps = float(cfg.sinkhorn_entropy_regularization)
    sim = torch.bmm(a, b.transpose(1, 2))
    batch, rows, cols = sim.shape
    log_alpha = sim.new_full((batch, rows + 1, cols + 1), float(cfg.sinkhorn_dustbin_score) / eps)
    log_alpha[:, :rows, :cols] = sim / eps
    total = 1.0 + float(cfg.sinkhorn_dustbin_mass)
    row_real_mass = 1.0 / (float(rows) * total)
    col_real_mass = 1.0 / (float(cols) * total)
    unmatched_mass = float(cfg.sinkhorn_dustbin_mass) / total
    row_mu = sim.new_full((batch, rows + 1), row_real_mass)
    col_nu = sim.new_full((batch, cols + 1), col_real_mass)
    row_mu[:, rows] = unmatched_mass
    col_nu[:, cols] = unmatched_mass
    log_mu = torch.log(row_mu)
    log_nu = torch.log(col_nu)
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)
    for _ in range(int(cfg.sinkhorn_iterations)):
        log_u = log_mu - torch.logsumexp(log_alpha + log_v[:, None, :], dim=2)
        log_v = log_nu - torch.logsumexp(log_alpha + log_u[:, :, None], dim=1)
    log_transport = log_alpha + log_u[:, :, None] + log_v[:, None, :]
    real_transport = torch.exp(log_transport[:, :rows, :cols])
    real_mass = real_transport.sum(dim=(1, 2)).clamp_min(torch.finfo(sim.dtype).eps)
    score = torch.sum(real_transport * sim, dim=(1, 2)) / real_mass
    if not return_diagnostics:
        return score, {}

    p = real_transport / real_mass[:, None, None]
    entropy = -torch.sum(p * torch.log(p.clamp_min(torch.finfo(p.dtype).eps)), dim=(1, 2))
    effective_matches = torch.exp(entropy)
    ya, xa = _positions(sim.device, rows)
    yb, xb = _positions(sim.device, cols)
    dy = yb.view(1, 1, cols) - ya.view(1, rows, 1)
    dx = xb.view(1, 1, cols) - xa.view(1, rows, 1)
    diagnostics = {
        "transport_real_mass": real_mass,
        "transport_weighted_cosine": score,
        "assignment_entropy": entropy,
        "assignment_entropy_normalized": entropy / math.log(max(2, rows * cols)),
        "effective_match_count": effective_matches,
        "mean_dx": torch.sum(p * dx, dim=(1, 2)),
        "mean_dy": torch.sum(p * dy, dim=(1, 2)),
        "mean_abs_dx": torch.sum(p * torch.abs(dx), dim=(1, 2)),
        "mean_abs_dy": torch.sum(p * torch.abs(dy), dim=(1, 2)),
        "dustbin_dominated": (real_mass < float(cfg.dustbin_dominated_real_mass_threshold)).float(),
    }
    return score, diagnostics


def _descriptor_pair_diagnostics(
    a: torch.Tensor,
    b: torch.Tensor,
    model_diag: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    combined = torch.cat([a, b], dim=1)
    flat = combined.reshape(combined.shape[0], -1, combined.shape[2])
    variance = torch.var(flat, dim=1, unbiased=False).mean(dim=1)
    sample = min(256, int(combined.shape[1]))
    if sample > 1:
        idx = torch.linspace(0, combined.shape[1] - 1, sample, device=combined.device).long()
        desc = combined.index_select(1, idx)
        sim = torch.bmm(desc, desc.transpose(1, 2))
        tri = torch.triu_indices(sample, sample, offset=1, device=combined.device)
        near = (sim[:, tri[0], tri[1]] > 0.9999).float().mean(dim=1)
    else:
        near = torch.zeros((combined.shape[0],), dtype=combined.dtype, device=combined.device)
    out = {
        "output_descriptor_variance": variance,
        "near_identical_context_descriptor_fraction": near,
        "pre_norm_a_mean": model_diag.get("pre_norm_a_mean", torch.full_like(variance, float("nan"))),
        "pre_norm_b_mean": model_diag.get("pre_norm_b_mean", torch.full_like(variance, float("nan"))),
        "pre_norm_a_std": model_diag.get("pre_norm_a_std", torch.full_like(variance, float("nan"))),
        "pre_norm_b_std": model_diag.get("pre_norm_b_std", torch.full_like(variance, float("nan"))),
        "self_attention_entropy": model_diag.get("self_attention_entropy", torch.full_like(variance, float("nan"))),
        "cross_attention_entropy": model_diag.get("cross_attention_entropy", torch.full_like(variance, float("nan"))),
    }
    return out


def condition_scores(
    *,
    condition: str,
    model: Optional[LocalCorrespondenceModel],
    a: torch.Tensor,
    b: torch.Tensor,
    cfg: PairConditionedConfig,
    need_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if condition == C0:
        za, zb = a.float(), b.float()
        model_diag: dict[str, torch.Tensor] = {
            "pre_norm_a_mean": torch.linalg.vector_norm(za, dim=2).mean(dim=1),
            "pre_norm_b_mean": torch.linalg.vector_norm(zb, dim=2).mean(dim=1),
            "pre_norm_a_std": torch.linalg.vector_norm(za, dim=2).std(dim=1, unbiased=False),
            "pre_norm_b_std": torch.linalg.vector_norm(zb, dim=2).std(dim=1, unbiased=False),
        }
    else:
        if model is None:
            raise PairConditionedError(f"{condition} requires a model")
        za, zb, model_diag = model.forward_pair(a, b, need_diagnostics=need_diagnostics)
    score, sink_diag = sinkhorn_scores_and_diagnostics(za, zb, cfg=cfg, return_diagnostics=need_diagnostics)
    if not need_diagnostics:
        return score, {}
    desc_diag = _descriptor_pair_diagnostics(za, zb, model_diag)
    return score, {**sink_diag, **desc_diag}


def _select_descriptors(
    tensors: local.TensorDescriptorCache,
    idx: Iterable[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    index = torch.as_tensor(list(idx), dtype=torch.long)
    return tensors.local_flat.index_select(0, index).to(device=device, dtype=torch.float32, non_blocking=False)


def autocast_for_device(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")


def grad_scaler_for_device(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled) and device.type == "cuda")


def _pair_indices_from_frame(df: pd.DataFrame, uid_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return local._pair_indices(df, uid_to_index)


@torch.no_grad()
def score_pair_frame_condition(
    *,
    condition: str,
    model: Optional[LocalCorrespondenceModel],
    df: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    cfg: PairConditionedConfig,
    need_diagnostics: bool,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    if model is not None:
        model.eval()
    scores = np.full(len(df), np.nan, dtype=np.float64)
    detail_rows: list[dict[str, Any]] = []
    idx_a, idx_b, valid = _pair_indices_from_frame(df, tensors.uid_to_index)
    valid_positions = np.flatnonzero(valid)
    start_time = time.perf_counter()
    for start in range(0, len(valid_positions), int(cfg.eval_pair_chunk_size)):
        pos = valid_positions[start : start + int(cfg.eval_pair_chunk_size)]
        a = _select_descriptors(tensors, idx_a[pos], device=device)
        b = _select_descriptors(tensors, idx_b[pos], device=device)
        with autocast_for_device(device, cfg.amp):
            score_t, diag_t = condition_scores(
                condition=condition,
                model=model,
                a=a,
                b=b,
                cfg=cfg,
                need_diagnostics=need_diagnostics,
            )
        score_np = score_t.detach().cpu().numpy().astype(np.float64)
        scores[pos] = score_np
        diag_np = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in diag_t.items()}
        for j, row_pos in enumerate(pos):
            row = df.iloc[int(row_pos)]
            payload = {
                "_row_order": int(row_pos),
                "condition": condition,
                "protocol": str(row.get("protocol_id", "")),
                "pair_id": str(row.get("pair_id", "")),
                "label": int(row.get("label", -1)),
                "sample_uid_a": str(row.get("sample_uid_a", "")),
                "sample_uid_b": str(row.get("sample_uid_b", "")),
                "finger_unit_a": str(row.get("finger_unit_a", "")),
                "finger_unit_b": str(row.get("finger_unit_b", "")),
                "score": float(score_np[j]),
                "failed": not math.isfinite(float(score_np[j])),
            }
            for key, values in diag_np.items():
                payload[key] = float(values[j])
            detail_rows.append(payload)
    for row_pos in np.flatnonzero(~valid):
        row = df.iloc[int(row_pos)]
        detail_rows.append(
            {
                "_row_order": int(row_pos),
                "condition": condition,
                "protocol": str(row.get("protocol_id", "")),
                "pair_id": str(row.get("pair_id", "")),
                "label": int(row.get("label", -1)),
                "sample_uid_a": str(row.get("sample_uid_a", "")),
                "sample_uid_b": str(row.get("sample_uid_b", "")),
                "score": float("nan"),
                "failed": True,
            }
        )
    details = pd.DataFrame(detail_rows)
    if not details.empty:
        details = details.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])
    return scores, details, float(time.perf_counter() - start_time)


def metric_row(
    *,
    condition: str,
    stage: str,
    seed: int,
    protocol: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    elapsed_seconds: float,
    epoch: Optional[int] = None,
) -> dict[str, Any]:
    row = matcher.metric_row_from_scores(
        method=condition,
        stage=stage,
        protocol=protocol,
        df=df,
        scores=scores,
        elapsed_seconds=elapsed_seconds,
    )
    row["condition"] = condition
    row["seed"] = int(seed)
    row["epoch"] = int(epoch) if epoch is not None else ""
    row.pop("method", None)
    return row


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else float("nan"))


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def aggregate_correspondence(details: pd.DataFrame, *, condition: str, stage: str, seed: int) -> list[dict[str, Any]]:
    if details.empty:
        return []
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        "transport_real_mass",
        "transport_weighted_cosine",
        "assignment_entropy",
        "assignment_entropy_normalized",
        "effective_match_count",
        "mean_dx",
        "mean_dy",
        "mean_abs_dx",
        "mean_abs_dy",
        "dustbin_dominated",
    ]
    for protocol, proto_df in details.groupby("protocol", sort=False):
        for label, group_name in ((1, "genuine"), (0, "impostor"), (-1, "all")):
            group = proto_df if label == -1 else proto_df[proto_df["label"].astype(int) == label]
            if group.empty:
                continue
            row = {
                "condition": condition,
                "stage": stage,
                "seed": int(seed),
                "protocol": protocol,
                "pair_group": group_name,
                "pair_count": int(len(group)),
            }
            for col in numeric_cols:
                values = pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                row[f"{col}_mean"] = _nanmean(values)
                row[f"{col}_std"] = _nanstd(values)
                row[f"{col}_median"] = _nanmedian(values)
            rows.append(row)
    return rows


def aggregate_attention(details: pd.DataFrame, *, condition: str, stage: str, seed: int) -> list[dict[str, Any]]:
    if details.empty:
        return []
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        "output_descriptor_variance",
        "near_identical_context_descriptor_fraction",
        "pre_norm_a_mean",
        "pre_norm_b_mean",
        "pre_norm_a_std",
        "pre_norm_b_std",
        "self_attention_entropy",
        "cross_attention_entropy",
    ]
    for protocol, proto_df in details.groupby("protocol", sort=False):
        row = {
            "condition": condition,
            "stage": stage,
            "seed": int(seed),
            "protocol": protocol,
            "pair_count": int(len(proto_df)),
        }
        for col in numeric_cols:
            values = pd.to_numeric(proto_df.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = _nanmean(values)
            row[f"{col}_std"] = _nanstd(values)
            row[f"{col}_median"] = _nanmedian(values)
        rows.append(row)
    return rows


def evaluate_condition(
    *,
    condition: str,
    stage: str,
    seed: int,
    model: Optional[LocalCorrespondenceModel],
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    cfg: PairConditionedConfig,
    epoch: Optional[int] = None,
    full_protocols: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    all_details: list[pd.DataFrame] = []
    protocols = base.CONTROL_PROTOCOLS if full_protocols else (CLCB,)
    for protocol in protocols:
        df = pair_bundle[protocol].reset_index(drop=True).copy()
        if "protocol_id" not in df.columns:
            df["protocol_id"] = protocol
        scores, details, elapsed = score_pair_frame_condition(
            condition=condition,
            model=model,
            df=df,
            tensors=tensors,
            device=device,
            cfg=cfg,
            need_diagnostics=full_protocols,
        )
        row = metric_row(
            condition=condition,
            stage=stage,
            seed=seed,
            protocol=protocol,
            df=df,
            scores=scores,
            elapsed_seconds=elapsed,
            epoch=epoch,
        )
        if protocol == CLCB:
            metric_rows.append(row)
        else:
            within_rows.append(row)
        if full_protocols:
            all_details.append(details)

    retrieval_rows: list[dict[str, Any]] = []
    if full_protocols:
        retrieval_rows = retrieval_metrics(
            condition=condition,
            stage=stage,
            seed=seed,
            model=model,
            table=retrieval_table,
            tensors=tensors,
            device=device,
            cfg=cfg,
            epoch=epoch,
        )
    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    return (
        metric_rows,
        retrieval_rows,
        within_rows,
        aggregate_correspondence(details_df, condition=condition, stage=stage, seed=seed),
        aggregate_attention(details_df, condition=condition, stage=stage, seed=seed),
    )


def retrieval_metrics(
    *,
    condition: str,
    stage: str,
    seed: int,
    model: Optional[LocalCorrespondenceModel],
    table: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    cfg: PairConditionedConfig,
    epoch: Optional[int],
) -> list[dict[str, Any]]:
    if table.empty:
        return []
    rows: list[dict[str, Any]] = []
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
                "finger_unit_a": ids[i],
                "finger_unit_b": ids[j],
                "protocol_id": direction,
            }
            for i, probe_uid in enumerate(probe)
            for j, gallery_uid in enumerate(gallery)
        ]
        scores, _details, _elapsed = score_pair_frame_condition(
            condition=condition,
            model=model,
            df=pd.DataFrame(pair_rows),
            tensors=tensors,
            device=device,
            cfg=cfg,
            need_diagnostics=False,
        )
        sim = scores.reshape(n, n)
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
                "epoch": int(epoch) if epoch is not None else "",
                "direction": direction,
                "identity_count": int(n),
                "recall_at_1": float(np.mean(ranks_np <= 1)),
                "recall_at_5": float(np.mean(ranks_np <= min(5, n))),
                "mrr": float(np.mean(1.0 / ranks_np)),
            }
        )
    return rows


def build_identity_pools(images: pd.DataFrame, identity_ids: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    return base.build_identity_pools(images, identity_ids)


def epoch_batches(
    pools: dict[str, dict[str, list[str]]],
    identity_ids: Iterable[str],
    *,
    batch_identities: int,
    epoch: int,
    seed: int,
    shuffled_identity: bool,
) -> list[dict[str, Any]]:
    return base.epoch_batches(
        pools,
        identity_ids,
        batch_identities=batch_identities,
        epoch=epoch,
        seed=seed,
        shuffled_identity=shuffled_identity,
    )


def _candidate_indices(anchor_index: int, batch_n: int, negatives_per_anchor: int) -> list[int]:
    if batch_n < 2:
        raise PairConditionedError("Need at least two identities per batch")
    count = min(int(negatives_per_anchor), batch_n - 1)
    return [int(anchor_index)] + [int((anchor_index + offset) % batch_n) for offset in range(1, count + 1)]


def _uids_to_indices(uids: list[str], uid_to_index: dict[str, int]) -> list[int]:
    missing = [uid for uid in uids if str(uid) not in uid_to_index]
    if missing:
        raise PairConditionedError(f"Descriptor cache missing {len(missing)} uid(s); first={missing[:3]}")
    return [int(uid_to_index[str(uid)]) for uid in uids]


def candidate_scores_for_direction(
    *,
    model: LocalCorrespondenceModel,
    anchor_desc: torch.Tensor,
    candidate_desc: torch.Tensor,
    cfg: PairConditionedConfig,
) -> torch.Tensor:
    batch_n = int(anchor_desc.shape[0])
    pair_a: list[torch.Tensor] = []
    pair_b: list[torch.Tensor] = []
    candidate_count = 1 + min(int(cfg.negatives_per_anchor), batch_n - 1)
    for i in range(batch_n):
        for j in _candidate_indices(i, batch_n, cfg.negatives_per_anchor):
            pair_a.append(anchor_desc[i])
            pair_b.append(candidate_desc[j])
    a = torch.stack(pair_a, dim=0)
    b = torch.stack(pair_b, dim=0)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(a.shape[0]), int(cfg.train_pair_chunk_size)):
        end = min(int(a.shape[0]), start + int(cfg.train_pair_chunk_size))
        score, _diag = condition_scores(
            condition=model.condition,
            model=model,
            a=a[start:end],
            b=b[start:end],
            cfg=cfg,
            need_diagnostics=False,
        )
        chunks.append(score)
    return torch.cat(chunks, dim=0).reshape(batch_n, candidate_count)


def training_epoch(
    *,
    model: LocalCorrespondenceModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tensors: local.TensorDescriptorCache,
    train_images: pd.DataFrame,
    train_ids: list[str],
    device: torch.device,
    cfg: PairConditionedConfig,
    epoch: int,
    seed: int,
    shuffled_identity: bool = False,
) -> tuple[float, dict[str, Any]]:
    model.train()
    pools = build_identity_pools(train_images, train_ids)
    batches = epoch_batches(
        pools,
        train_ids,
        batch_identities=cfg.batch_identities,
        epoch=epoch,
        seed=seed,
        shuffled_identity=shuffled_identity,
    )
    losses: list[float] = []
    grad_check: dict[str, Any] = {}
    for batch in batches:
        cl_idx = _uids_to_indices([str(x) for x in batch["cl_uids"]], tensors.uid_to_index)
        cb_idx = _uids_to_indices([str(x) for x in batch["cb_uids"]], tensors.uid_to_index)
        cl_desc = _select_descriptors(tensors, cl_idx, device=device)
        cb_desc = _select_descriptors(tensors, cb_idx, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for_device(device, cfg.amp):
            cl_to_cb = candidate_scores_for_direction(model=model, anchor_desc=cl_desc, candidate_desc=cb_desc, cfg=cfg)
            cb_to_cl = candidate_scores_for_direction(model=model, anchor_desc=cb_desc, candidate_desc=cl_desc, cfg=cfg)
            target = torch.zeros((cl_to_cb.shape[0],), dtype=torch.long, device=device)
            loss = 0.5 * (
                F.cross_entropy(cl_to_cb / float(cfg.train_temperature), target)
                + F.cross_entropy(cb_to_cl / float(cfg.train_temperature), target)
            )
        if not torch.isfinite(loss):
            raise PairConditionedError(f"Non-finite training loss at epoch {epoch}: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        grad_check = gradient_status(model)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return (float(np.mean(losses)) if losses else float("nan")), grad_check


def train_condition(
    *,
    condition: str,
    stage: str,
    seed: int,
    tensors: local.TensorDescriptorCache,
    train_images: pd.DataFrame,
    train_ids: list[str],
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    device: torch.device,
    cfg: PairConditionedConfig,
    max_epochs: int,
    patience: int,
    encoder_sha256: str,
    shuffled_identity: bool = False,
) -> TrainRunResult:
    set_seed(seed)
    model = LocalCorrespondenceModel(
        condition=condition,
        projection_dim=cfg.projection_dim,
        attention_heads=cfg.attention_heads,
    ).to(device)
    trainable_param_count, total_param_count = count_params(model)
    names = trainable_names(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scaler = grad_scaler_for_device(device, cfg.amp)
    best_auc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    curve_rows: list[dict[str, Any]] = []
    epochs_without_gain = 0
    last_grad: dict[str, Any] = {}
    for epoch in range(1, int(max_epochs) + 1):
        train_loss, last_grad = training_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            tensors=tensors,
            train_images=train_images,
            train_ids=train_ids,
            device=device,
            cfg=cfg,
            epoch=epoch,
            seed=seed,
            shuffled_identity=shuffled_identity,
        )
        metric_rows, _retrieval, _within, _corr, _attn = evaluate_condition(
            condition=condition if not shuffled_identity else C2_SHUFFLED,
            stage=stage,
            seed=seed,
            model=model,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            tensors=tensors,
            device=device,
            cfg=cfg,
            epoch=epoch,
            full_protocols=False,
        )
        auc = float(metric_rows[0]["roc_auc"]) if metric_rows else float("nan")
        curve_rows.append(
            {
                "condition": condition if not shuffled_identity else C2_SHUFFLED,
                "stage": stage,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": train_loss,
                "inner_dev_clcb_auc": auc,
                "inner_dev_clcb_eer": float(metric_rows[0]["eer"]) if metric_rows else float("nan"),
                "finite_loss": bool(math.isfinite(train_loss)),
                "gradients_only_in_intended_components": bool(last_grad.get("gradients_only_in_intended_components", False)),
            }
        )
        if math.isfinite(auc) and auc > best_auc + 1e-9:
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= int(patience):
                break
    model.load_state_dict(best_state, strict=True)
    output_condition = condition if not shuffled_identity else C2_SHUFFLED
    metric_rows, retrieval_rows, within_rows, corr_rows, attn_rows = evaluate_condition(
        condition=output_condition,
        stage=stage,
        seed=seed,
        model=model,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        tensors=tensors,
        device=device,
        cfg=cfg,
        epoch=best_epoch,
        full_protocols=True,
    )
    return TrainRunResult(
        condition=output_condition,
        stage=stage,
        seed=seed,
        best_epoch=int(best_epoch),
        best_auc=float(best_auc),
        best_state_dict=best_state,
        curve_rows=curve_rows,
        metric_rows=metric_rows,
        retrieval_rows=retrieval_rows,
        within_rows=within_rows,
        correspondence_rows=corr_rows,
        attention_rows=attn_rows,
        trainable_param_count=trainable_param_count,
        total_param_count=total_param_count,
        trainable_names=names,
        gradient_check=last_grad,
        encoder_sha256_before=encoder_sha256,
        encoder_sha256_after=encoder_sha256,
    )


def evaluate_untrained_c2(
    *,
    seed: int,
    stage: str,
    tensors: local.TensorDescriptorCache,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    device: torch.device,
    cfg: PairConditionedConfig,
    encoder_sha256: str,
) -> TrainRunResult:
    set_seed(seed)
    model = LocalCorrespondenceModel(
        condition=C2,
        projection_dim=cfg.projection_dim,
        attention_heads=cfg.attention_heads,
    ).to(device)
    trainable_param_count, total_param_count = count_params(model)
    metric_rows, retrieval_rows, within_rows, corr_rows, attn_rows = evaluate_condition(
        condition=C2_RANDOM,
        stage=stage,
        seed=seed,
        model=model,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        tensors=tensors,
        device=device,
        cfg=cfg,
        epoch=0,
        full_protocols=True,
    )
    return TrainRunResult(
        condition=C2_RANDOM,
        stage=stage,
        seed=seed,
        best_epoch=0,
        best_auc=float(metric_rows[0]["roc_auc"]) if metric_rows else float("nan"),
        best_state_dict=copy.deepcopy(model.state_dict()),
        curve_rows=[],
        metric_rows=metric_rows,
        retrieval_rows=retrieval_rows,
        within_rows=within_rows,
        correspondence_rows=corr_rows,
        attention_rows=attn_rows,
        trainable_param_count=trainable_param_count,
        total_param_count=total_param_count,
        trainable_names=trainable_names(model),
        gradient_check={"gradients_only_in_intended_components": None, "note": "Untrained evaluation control."},
        encoder_sha256_before=encoder_sha256,
        encoder_sha256_after=encoder_sha256,
    )


def evaluate_c0(
    *,
    tensors: local.TensorDescriptorCache,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    device: torch.device,
    cfg: PairConditionedConfig,
    encoder_sha256: str,
) -> TrainRunResult:
    metric_rows, retrieval_rows, within_rows, corr_rows, attn_rows = evaluate_condition(
        condition=C0,
        stage="frozen_reference",
        seed=0,
        model=None,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        tensors=tensors,
        device=device,
        cfg=cfg,
        epoch=None,
        full_protocols=True,
    )
    return TrainRunResult(
        condition=C0,
        stage="frozen_reference",
        seed=0,
        best_epoch=0,
        best_auc=float(metric_rows[0]["roc_auc"]) if metric_rows else float("nan"),
        best_state_dict={},
        curve_rows=[],
        metric_rows=metric_rows,
        retrieval_rows=retrieval_rows,
        within_rows=within_rows,
        correspondence_rows=corr_rows,
        attention_rows=attn_rows,
        trainable_param_count=0,
        total_param_count=0,
        trainable_names=[],
        gradient_check={"gradients_only_in_intended_components": True, "note": "Frozen no-training reference."},
        encoder_sha256_before=encoder_sha256,
        encoder_sha256_after=encoder_sha256,
    )


def _row_auc(rows: list[dict[str, Any]], protocol: str = CLCB) -> float:
    for row in rows:
        if str(row.get("protocol")) == protocol:
            return float(row.get("roc_auc", float("nan")))
    return float("nan")


def _retrieval_mrr(rows: list[dict[str, Any]], direction: str) -> float:
    for row in rows:
        if str(row.get("direction")) == direction:
            return float(row.get("mrr", float("nan")))
    return float("nan")


def _within_auc_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row.get("protocol")): float(row.get("roc_auc", float("nan"))) for row in rows}


def _attention_summary(attention_rows: list[dict[str, Any]]) -> dict[str, float]:
    variances = [float(row.get("output_descriptor_variance_mean", float("nan"))) for row in attention_rows]
    near = [float(row.get("near_identical_context_descriptor_fraction_mean", float("nan"))) for row in attention_rows]
    return {
        "output_descriptor_variance_mean_min": float(np.nanmin(variances)) if np.isfinite(variances).any() else float("nan"),
        "near_identical_context_descriptor_fraction_max": float(np.nanmax(near)) if np.isfinite(near).any() else float("nan"),
    }


def signal_gate_for_result(
    result: TrainRunResult,
    *,
    c0: TrainRunResult,
    cfg: PairConditionedConfig,
) -> dict[str, Any]:
    auc = _row_auc(result.metric_rows)
    c0_within = _within_auc_map(c0.within_rows)
    within = _within_auc_map(result.within_rows)
    cl_mrr = _retrieval_mrr(result.retrieval_rows, "CL_probe_to_CB_gallery")
    cb_mrr = _retrieval_mrr(result.retrieval_rows, "CB_probe_to_CL_gallery")
    attn_summary = _attention_summary(result.attention_rows)
    no_collapse = bool(
        math.isfinite(attn_summary["output_descriptor_variance_mean_min"])
        and attn_summary["output_descriptor_variance_mean_min"] >= cfg.no_collapse_variance_min
        and (
            not math.isfinite(attn_summary["near_identical_context_descriptor_fraction_max"])
            or attn_summary["near_identical_context_descriptor_fraction_max"] <= cfg.no_collapse_near_identical_max
        )
    )
    within_details: list[dict[str, Any]] = []
    for protocol in WITHIN_PROTOCOLS:
        observed = float(within.get(protocol, float("nan")))
        reference = float(c0_within.get(protocol, float("nan")))
        threshold = max(float(cfg.within_min_auc_floor), reference - float(cfg.within_drop_tolerance_vs_c0))
        within_details.append(
            {
                "protocol": protocol,
                "auc": observed,
                "c0_reference_auc": reference,
                "threshold": threshold,
                "passed": bool(math.isfinite(observed) and observed >= threshold),
            }
        )
    criteria = {
        "inner_dev_clcb_auc_at_least_threshold": bool(math.isfinite(auc) and auc >= cfg.signal_auc_threshold),
        "auc_gain_over_hard_mnn_at_least_threshold": bool(
            math.isfinite(auc)
            and (auc - BASELINE_HARD_MNN_F_RAW_AUC) >= cfg.signal_auc_gain_over_hard_mnn
        ),
        "cl_to_cb_retrieval_mrr_improves": bool(
            math.isfinite(cl_mrr)
            and cl_mrr > BASELINE_HARD_MNN_RETRIEVAL["CL_probe_to_CB_gallery"]["mrr"] + cfg.retrieval_mrr_gain
        ),
        "cb_to_cl_retrieval_mrr_improves": bool(
            math.isfinite(cb_mrr)
            and cb_mrr > BASELINE_HARD_MNN_RETRIEVAL["CB_probe_to_CL_gallery"]["mrr"] + cfg.retrieval_mrr_gain
        ),
        "no_descriptor_collapse": no_collapse,
        "within_modality_controls_do_not_materially_collapse": all(bool(x["passed"]) for x in within_details),
        "encoder_sha256_unchanged": result.encoder_sha256_before == result.encoder_sha256_after,
        "gradients_only_in_intended_components": bool(
            result.gradient_check.get("gradients_only_in_intended_components", True)
        ),
    }
    return {
        "condition": result.condition,
        "stage": result.stage,
        "seed": int(result.seed),
        "best_epoch": int(result.best_epoch),
        "inner_dev_clcb_auc": auc,
        "auc_gain_over_hard_mnn_f_raw": auc - BASELINE_HARD_MNN_F_RAW_AUC if math.isfinite(auc) else float("nan"),
        "cl_to_cb_mrr": cl_mrr,
        "cb_to_cl_mrr": cb_mrr,
        "attention_collapse_summary": attn_summary,
        "within_modality_details": within_details,
        "criteria": criteria,
        "passed": all(bool(v) for v in criteria.values()),
    }


def aggregate_seed_results(results: list[TrainRunResult], c0: TrainRunResult, cfg: PairConditionedConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in sorted({r.condition for r in results}):
        values = [signal_gate_for_result(r, c0=c0, cfg=cfg)["inner_dev_clcb_auc"] for r in results if r.condition == condition]
        finite = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
        rows.append(
            {
                "condition": condition,
                "seed_count": int(len(values)),
                "auc_mean": float(np.mean(finite)) if finite.size else float("nan"),
                "auc_median": float(np.median(finite)) if finite.size else float("nan"),
                "auc_std": float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else float("nan")),
                "auc_min": float(np.min(finite)) if finite.size else float("nan"),
                "auc_max": float(np.max(finite)) if finite.size else float("nan"),
                "passing_seed_count": int(
                    sum(signal_gate_for_result(r, c0=c0, cfg=cfg)["passed"] for r in results if r.condition == condition)
                ),
            }
        )
    return rows


def classify_decision(
    *,
    stage2_results: list[TrainRunResult],
    stage3_results: list[TrainRunResult],
    c0: TrainRunResult,
    cfg: PairConditionedConfig,
) -> dict[str, Any]:
    stage2_gate = [signal_gate_for_result(r, c0=c0, cfg=cfg) for r in stage2_results]
    passing_stage2 = [row["condition"] for row in stage2_gate if row["passed"]]
    stage3_gate = [signal_gate_for_result(r, c0=c0, cfg=cfg) for r in stage3_results]
    stable_counts: dict[str, int] = {}
    for row in stage3_gate:
        stable_counts[row["condition"]] = stable_counts.get(row["condition"], 0) + int(bool(row["passed"]))
    stable_conditions = [cond for cond, count in stable_counts.items() if count >= 2]
    aggregates = aggregate_seed_results(stage3_results, c0, cfg) if stage3_results else []

    def _median_auc(condition: str, fallback: float = float("nan")) -> float:
        for row in aggregates:
            if row["condition"] == condition:
                return float(row["auc_median"])
        for row in stage2_gate:
            if row["condition"] == condition:
                return float(row["inner_dev_clcb_auc"])
        return fallback

    c1_stable = C1 in stable_conditions
    c2_stable = C2 in stable_conditions
    c2_stage2_passed = C2 in passing_stage2
    c1_auc = _median_auc(C1)
    c2_auc = _median_auc(C2)
    c2_material = bool(math.isfinite(c2_auc) and math.isfinite(c1_auc) and c2_auc >= c1_auc + cfg.material_c2_auc_gain_over_c1)
    if c2_stable and c2_material:
        classification = "A. PAIR_CONDITIONING_SUFFICIENT"
        reason = "C2 passed stability and materially outperformed C1."
    elif c1_stable and not c2_material:
        classification = "B. PROJECTION_SUFFICIENT"
        reason = "C1 passed stability and C2 was not materially better."
    elif c2_stage2_passed and not c2_stable:
        classification = "C. PAIR_CONDITIONING_SIGNAL_BUT_UNSTABLE"
        reason = "C2 passed the single-seed signal gate but did not pass stability."
    else:
        classification = "D. PAIR_CONDITIONED_LOCAL_ALIGNMENT_INSUFFICIENT"
        reason = "C1 and C2 failed the predeclared signal gate."
    return {
        "classification": classification,
        "primary_reason": reason,
        "stage2_signal_gate": stage2_gate,
        "stage3_multiseed": {
            "ran": bool(stage3_results),
            "passing_conditions_from_stage2": passing_stage2,
            "seed_gate": stage3_gate,
            "stable_pass_counts": stable_counts,
            "stable_conditions": stable_conditions,
            "aggregate": aggregates,
        },
        "official_val_gate": {
            "opened": False,
            "reason": (
                "Official VAL remains closed because stability gate did not pass."
                if not stable_conditions
                else "Stability gate passed; official VAL branch is intentionally not executed by this diagnostic runner."
            ),
        },
        "test_gate": {"opened": False, "reason": "TEST is prohibited for Phase 4B.3A."},
    }


def experiment_config(cfg: PairConditionedConfig, *, checkpoint: Path, phase4b1_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_required": DEFAULT_CHECKPOINT_SHA256,
        "inner_split_source": str(Path(phase4b1_dir) / "inner_split.json"),
        "preprocess_contract": base.PREPROCESS_CONTRACT,
        "feature_map": {"stage": local.SELECTED_STAGE_NAME, "shape": [24, 24, 256]},
        "conditions": {
            C0: "Frozen F-stage descriptors scored by Phase 4B.2B.1 S3 Sinkhorn partial assignment.",
            C1: "Frozen encoder; shared Linear(256,128); per-descriptor L2 normalization; same Sinkhorn head.",
            C2: (
                "Frozen encoder; shared Linear(256,128); fixed 2D sin/cos positional encoding; one block with "
                "self-attention in each image and bidirectional cross-attention; residual LayerNorm; L2 normalization; "
                "same Sinkhorn head."
            ),
        },
        "attention": {"model_dim": cfg.projection_dim, "heads": cfg.attention_heads, "blocks": cfg.attention_blocks},
        "sinkhorn": {
            "entropy_regularization": cfg.sinkhorn_entropy_regularization,
            "iterations": cfg.sinkhorn_iterations,
            "dustbin_score": cfg.sinkhorn_dustbin_score,
            "dustbin_mass": cfg.sinkhorn_dustbin_mass,
        },
        "training_candidates": {
            "batch_identities": cfg.batch_identities,
            "negatives_per_anchor": cfg.negatives_per_anchor,
            "negative_policy": "For anchor i, candidates are positive index i then deterministic offsets i+1..i+K modulo batch size.",
            "directions": ["CL_anchor_to_CB_candidates", "CB_anchor_to_CL_candidates"],
            "temperature": cfg.train_temperature,
        },
        "training_funnel": {
            "stage1_smoke": ["C1 seed 13 one epoch", "C2 seed 13 one epoch"],
            "stage2_signal": ["C1 seed 13 max 15 patience 4", "C2 seed 13 max 15 patience 4"],
            "stage3_only_if_signal": list(cfg.stage3_seeds),
        },
        "guardrails": {
            "test_read": False,
            "official_val_for_development": False,
            "encoder_frozen_for_c1_c2": True,
            "p2_used": False,
            "metadata_inputs": False,
            "broad_hyperparameter_search": False,
        },
    }


def _unique_uids_for_training_and_eval(
    train_images: pd.DataFrame,
    eval_images: pd.DataFrame,
    pair_bundle: dict[str, pd.DataFrame],
) -> list[str]:
    uids = set(train_images["sample_uid"].astype(str).tolist())
    uids.update(eval_images["sample_uid"].astype(str).tolist())
    for df in pair_bundle.values():
        uids.update(df["sample_uid_a"].astype(str).tolist())
        uids.update(df["sample_uid_b"].astype(str).tolist())
    return sorted(uids)


def run(
    *,
    manifest_dir: Path,
    phase4b1_dir: Path,
    checkpoint: Path,
    outdir: Path,
    polyu_root: Optional[str],
    device_arg: str,
    cfg: PairConditionedConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    if cfg.attention_blocks != 1:
        raise PairConditionedError("Phase 4B.3A permits exactly one attention block")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    write_json(outdir / "experiment_config.json", experiment_config(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir))
    device = local.resolve_device(device_arg)
    set_seed(cfg.seed)

    train_images_all, resolved_root = local.load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[base.TRAIN])
    official_train_ids = sorted(train_images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = local.load_fixed_inner_split(official_train_ids, phase4b1_dir, split_seed=cfg.split_seed)
    inner_train_ids = inner_split["inner_train"]
    inner_dev_ids = inner_split["inner_dev"]
    if smoke:
        inner_train_ids = inner_train_ids[: min(24, len(inner_train_ids))]
        inner_dev_ids = inner_dev_ids[: min(8, len(inner_dev_ids))]
    train_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_train_ids))].copy()
    inner_dev_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    pair_max_pos = min(cfg.eval_max_pos, 40) if smoke else cfg.eval_max_pos
    pair_neg_per_pos = min(cfg.eval_neg_per_pos, 1) if smoke else cfg.eval_neg_per_pos
    stage2_max_epochs = min(2, cfg.max_epochs) if smoke else cfg.max_epochs
    stage2_patience = 1 if smoke else cfg.patience

    pair_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=pair_max_pos,
        neg_per_pos=pair_neg_per_pos,
        seed=cfg.split_seed,
    )
    pair_counts = base.validate_pair_bundle(pair_bundle, stage="inner_dev")
    retrieval_table = base.build_retrieval_table(inner_dev_images, inner_dev_ids)

    pair_model, ckpt_args, checkpoint_meta, checkpoint_sha = local.load_frozen_pair_model(
        checkpoint=checkpoint,
        device=device,
        require_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    encoder_sha_before = learned.state_sha256(pair_model.state_dict(), prefix="encoder.")
    input_size = int(ckpt_args.get("input_size", 384))
    all_images = pd.concat([train_images, inner_dev_images], axis=0).drop_duplicates("sample_uid")
    image_store = local.load_image_store_for_rows(all_images, input_size=input_size)
    uids = _unique_uids_for_training_and_eval(train_images, inner_dev_images, pair_bundle)
    descriptor_cache = local.extract_descriptor_cache(
        encoder=pair_model.encoder,
        image_store=image_store,
        uids=uids,
        device=device,
        batch_size=cfg.eval_batch_size,
        selected_stage_index=local.SELECTED_STAGE_INDEX,
        amp=cfg.amp,
    )
    encoder_sha_after = learned.state_sha256(pair_model.state_dict(), prefix="encoder.")
    if encoder_sha_before != encoder_sha_after:
        raise PairConditionedError("Encoder SHA changed during frozen descriptor extraction")
    tensors = local.descriptor_cache_to_tensors(descriptor_cache, torch.device("cpu"))
    del image_store
    pair_model.to(torch.device("cpu"))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[TrainRunResult] = []
    print("[eval] C0 frozen Sinkhorn reference", flush=True)
    c0 = evaluate_c0(tensors=tensors, pair_bundle=pair_bundle, retrieval_table=retrieval_table, device=device, cfg=cfg, encoder_sha256=encoder_sha_before)
    results.append(c0)
    print("[eval] C2 random initialization control", flush=True)
    results.append(
        evaluate_untrained_c2(
            seed=cfg.seed,
            stage="random_initialization_control",
            tensors=tensors,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            device=device,
            cfg=cfg,
            encoder_sha256=encoder_sha_before,
        )
    )

    stage1_results: list[TrainRunResult] = []
    for condition in (C1, C2):
        print(f"[stage1] {condition}", flush=True)
        result = train_condition(
            condition=condition,
            stage="stage1_smoke",
            seed=cfg.seed,
            tensors=tensors,
            train_images=train_images,
            train_ids=inner_train_ids,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            device=device,
            cfg=cfg,
            max_epochs=1,
            patience=1,
            encoder_sha256=encoder_sha_before,
        )
        stage1_results.append(result)
        results.append(result)

    print("[control] C2 shuffled identity", flush=True)
    shuffled = train_condition(
        condition=C2,
        stage="shuffled_identity_control",
        seed=stable_int(cfg.seed, "shuffled") % (2**31),
        tensors=tensors,
        train_images=train_images,
        train_ids=inner_train_ids,
        pair_bundle=pair_bundle,
        retrieval_table=retrieval_table,
        device=device,
        cfg=cfg,
        max_epochs=1,
        patience=1,
        encoder_sha256=encoder_sha_before,
        shuffled_identity=True,
    )
    results.append(shuffled)

    stage2_results: list[TrainRunResult] = []
    for condition in (C1, C2):
        print(f"[stage2] {condition}", flush=True)
        result = train_condition(
            condition=condition,
            stage="stage2_signal",
            seed=cfg.seed,
            tensors=tensors,
            train_images=train_images,
            train_ids=inner_train_ids,
            pair_bundle=pair_bundle,
            retrieval_table=retrieval_table,
            device=device,
            cfg=cfg,
            max_epochs=stage2_max_epochs,
            patience=stage2_patience,
            encoder_sha256=encoder_sha_before,
        )
        stage2_results.append(result)
        results.append(result)

    stage2_gate = [signal_gate_for_result(r, c0=c0, cfg=cfg) for r in stage2_results]
    passing = [row["condition"] for row in stage2_gate if row["passed"]]
    stage3_results: list[TrainRunResult] = []
    if passing and not smoke:
        for condition in sorted(set(passing)):
            for seed in cfg.stage3_seeds:
                existing = next((r for r in stage2_results if r.condition == condition and r.seed == seed), None)
                if existing is not None:
                    stage3_results.append(existing)
                    continue
                print(f"[stage3] {condition} seed={seed}", flush=True)
                result = train_condition(
                    condition=condition,
                    stage="stage3_multiseed",
                    seed=int(seed),
                    tensors=tensors,
                    train_images=train_images,
                    train_ids=inner_train_ids,
                    pair_bundle=pair_bundle,
                    retrieval_table=retrieval_table,
                    device=device,
                    cfg=cfg,
                    max_epochs=cfg.max_epochs,
                    patience=cfg.patience,
                    encoder_sha256=encoder_sha_before,
                )
                stage3_results.append(result)
                results.append(result)

    decision = classify_decision(stage2_results=stage2_results, stage3_results=stage3_results, c0=c0, cfg=cfg)
    decision["stage1_smoke"] = {
        "conditions": [
            {
                "condition": r.condition,
                "seed": int(r.seed),
                "best_epoch": int(r.best_epoch),
                "inner_dev_clcb_auc": _row_auc(r.metric_rows),
                "finite_losses": all(math.isfinite(float(row["train_loss"])) for row in r.curve_rows),
                "encoder_sha256_unchanged": r.encoder_sha256_before == r.encoder_sha256_after,
                "gradient_check": r.gradient_check,
            }
            for r in stage1_results
        ]
    }

    metric_rows = [row for r in results for row in r.metric_rows]
    retrieval_rows = [row for r in results for row in r.retrieval_rows]
    within_rows = [row for r in results for row in r.within_rows]
    corr_rows = [row for r in results for row in r.correspondence_rows]
    attn_rows = [row for r in results for row in r.attention_rows]
    curve_rows = [row for r in results for row in r.curve_rows]
    inventory_rows = [
        {
            "condition": r.condition,
            "stage": r.stage,
            "seed": int(r.seed),
            "best_epoch": int(r.best_epoch),
            "best_inner_dev_clcb_auc": _row_auc(r.metric_rows),
            "trainable_param_count": int(r.trainable_param_count),
            "total_param_count": int(r.total_param_count),
            "trainable_names": json.dumps(r.trainable_names),
            "encoder_sha256_before": r.encoder_sha256_before,
            "encoder_sha256_after": r.encoder_sha256_after,
            "encoder_sha256_unchanged": r.encoder_sha256_before == r.encoder_sha256_after,
        }
        for r in results
    ]

    write_csv(outdir / "condition_inventory.csv", inventory_rows)
    write_csv(outdir / "training_curves.csv", curve_rows)
    write_csv(outdir / "inner_dev_metrics.csv", metric_rows)
    write_csv(outdir / "retrieval_metrics.csv", retrieval_rows)
    write_csv(outdir / "within_modality_controls.csv", within_rows)
    write_csv(outdir / "correspondence_diagnostics.csv", corr_rows)
    write_csv(outdir / "attention_diagnostics.csv", attn_rows)
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
        "preprocess_contract": base.PREPROCESS_CONTRACT,
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "inner_split": {
            "source": str(Path(phase4b1_dir) / "inner_split.json"),
            "official_train_identity_count": int(len(official_train_ids)),
            "inner_train_identity_count": int(len(inner_train_ids)),
            "inner_dev_identity_count": int(len(inner_dev_ids)),
            "identity_disjoint": not bool(set(inner_train_ids).intersection(inner_dev_ids)),
            "smoke": bool(smoke),
        },
        "pair_counts": {"inner_dev": pair_counts},
        "feature_map": {
            "stage_name": descriptor_cache.selected_stage_name,
            "height": int(descriptor_cache.height),
            "width": int(descriptor_cache.width),
            "channels": int(descriptor_cache.channels),
        },
        "config": experiment_config(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir),
        "decision": decision,
        "outputs": {
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "condition_inventory_csv": str(outdir / "condition_inventory.csv"),
            "training_curves_csv": str(outdir / "training_curves.csv"),
            "inner_dev_metrics_csv": str(outdir / "inner_dev_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "within_modality_controls_csv": str(outdir / "within_modality_controls.csv"),
            "correspondence_diagnostics_csv": str(outdir / "correspondence_diagnostics.csv"),
            "attention_diagnostics_csv": str(outdir / "attention_diagnostics.csv"),
            "signal_gate_decision_json": str(outdir / "signal_gate_decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
        },
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
        "canonical_artifacts_not_read": {
            "pairs_val_csv": str(Path(manifest_dir) / "pairs_val.csv"),
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "Official VAL remains gated and TEST remains closed for Phase 4B.3A.",
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
            "official_val_read_for_development": False,
            "official_val_gate_opened": False,
            "canonical_manifest_or_pairs_modified": False,
            "canonical_checkpoint_modified": False,
            "encoder_frozen_for_c1_c2": True,
            "encoder_sha256_before": encoder_sha_before,
            "encoder_sha256_after": encoder_sha_after,
            "encoder_sha256_unchanged": encoder_sha_before == encoder_sha_after,
            "used_p2": False,
            "used_fusion_or_sourceafis_targets": False,
            "used_sift_targets": False,
            "used_metadata_as_model_input": False,
            "trained_transformer_backbone": False,
            "attention_blocks": int(cfg.attention_blocks),
            "implemented_ridgeformer": False,
            "geometric_unwarping": False,
            "broad_architecture_or_hyperparameter_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "decision": decision,
        "inventory": pd.DataFrame(inventory_rows),
        "metrics": pd.DataFrame(metric_rows),
        "retrieval": pd.DataFrame(retrieval_rows),
        "within": pd.DataFrame(within_rows),
        "correspondence": pd.DataFrame(corr_rows),
        "attention": pd.DataFrame(attn_rows),
        "run_manifest": run_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.3A pair-conditioned local correspondence learning for PolyU Cross.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_epochs", type=int, default=PairConditionedConfig.max_epochs)
    p.add_argument("--patience", type=int, default=PairConditionedConfig.patience)
    p.add_argument("--eval_batch_size", type=int, default=PairConditionedConfig.eval_batch_size)
    p.add_argument("--train_pair_chunk_size", type=int, default=PairConditionedConfig.train_pair_chunk_size)
    p.add_argument("--eval_pair_chunk_size", type=int, default=PairConditionedConfig.eval_pair_chunk_size)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PairConditionedConfig(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        eval_batch_size=int(args.eval_batch_size),
        train_pair_chunk_size=int(args.train_pair_chunk_size),
        eval_pair_chunk_size=int(args.eval_pair_chunk_size),
        amp=bool(args.amp),
    )
    try:
        result = run(
            manifest_dir=resolve_repo_path(args.data_dir),
            phase4b1_dir=resolve_repo_path(args.phase4b1_dir),
            checkpoint=resolve_repo_path(args.checkpoint),
            outdir=resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            device_arg=str(args.device),
            cfg=cfg,
            smoke=bool(args.smoke),
        )
    except (PairConditionedError, local.LocalCorrespondenceError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    decision = result["decision"]
    print("\n=== PolyU Cross Phase 4B.3A pair-conditioned correspondence complete ===")
    print(f"Output dir     : {result['outdir']}")
    print(f"Classification : {decision['classification']}")
    print(f"Official VAL   : {'opened' if decision['official_val_gate']['opened'] else 'closed'}")
    print("TEST           : closed")
    show = result["metrics"][["condition", "stage", "seed", "epoch", "roc_auc", "eer"]]
    print("\nInner-dev CL->CB:")
    print(show.to_string(index=False))
    print("\nRetrieval:")
    print(result["retrieval"][["condition", "stage", "direction", "recall_at_1", "recall_at_5", "mrr"]].to_string(index=False))
    print("\nWithin controls:")
    within = result["within"].pivot_table(index=["condition", "stage"], columns="protocol", values="roc_auc", aggfunc="first").reset_index()
    print(within.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
