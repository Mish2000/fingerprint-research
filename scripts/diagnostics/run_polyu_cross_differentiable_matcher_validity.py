"""Phase 4B.2B.1 differentiable local-matcher validity audit for PolyU Cross.

This diagnostic audits local matching functions on frozen ConvEncoder F-stage
descriptors only. It deliberately performs no training, creates no optimizer,
does not use P2, never opens official VAL, and never reads TEST.
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
import torch.nn.functional as F

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, safe_pkg_version, sha256_file, utc_now
from scripts.diagnostics import run_polyu_cross_learned_local_alignment as learned
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import PolyUCrossPairError


RUN_SCHEMA_VERSION = "polyu_cross_differentiable_matcher_validity_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_differentiable_matcher_validity_v0"
DEFAULT_CHECKPOINT_SHA256 = local.DEFAULT_CHECKPOINT_SHA256

S0_HARD_MNN_REFERENCE = "S0_hard_mnn_reference"
S1_CURRENT_TOPK_MAX = "S1_current_topk_max"
S2_DUAL_SOFTMAX = "S2_dual_softmax"
S3_SINKHORN_PARTIAL_ASSIGNMENT = "S3_sinkhorn_partial_assignment"
ALL_SCORERS = (
    S0_HARD_MNN_REFERENCE,
    S1_CURRENT_TOPK_MAX,
    S2_DUAL_SOFTMAX,
    S3_SINKHORN_PARTIAL_ASSIGNMENT,
)
DIFFERENTIABLE_SCORERS = (S1_CURRENT_TOPK_MAX, S2_DUAL_SOFTMAX, S3_SINKHORN_PARTIAL_ASSIGNMENT)
PRIMARY_DIFFERENTIABLE_SCORERS = (S2_DUAL_SOFTMAX, S3_SINKHORN_PARTIAL_ASSIGNMENT)

CLCB = "contactless_to_contact_based"
CLCL_SAME = "contactless_to_contactless_same_session"
CLCL_CROSS = "contactless_to_contactless_cross_session"
CBCB_SAME = "contact_based_to_contact_based_same_session"
CBCB_CROSS = "contact_based_to_contact_based_cross_session"

EXPECTED_REFERENCE_AUC = {
    S0_HARD_MNN_REFERENCE: {
        CLCB: 0.5119,
        CLCL_SAME: 0.9723,
        CLCL_CROSS: 0.7826,
        CBCB_SAME: 0.9192,
        CBCB_CROSS: 0.7363,
    },
    S1_CURRENT_TOPK_MAX: {
        CLCB: 0.4946,
        CLCL_SAME: 0.9061,
        CLCL_CROSS: 0.6907,
        CBCB_SAME: 0.5230,
        CBCB_CROSS: 0.5129,
    },
}


class MatcherValidityError(RuntimeError):
    """Raised for Phase 4B.2B.1 protocol or artifact failures."""


@dataclass(frozen=True)
class MatcherAuditConfig:
    seed: int = 1341
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 64
    score_batch_size: int = 8
    selected_stage_index: int = local.SELECTED_STAGE_INDEX
    topk_fraction: float = 0.10
    dual_softmax_temperature: float = 0.07
    sinkhorn_entropy_regularization: float = 0.07
    sinkhorn_iterations: int = 20
    sinkhorn_dustbin_score: float = 0.0
    sinkhorn_dustbin_mass: float = 1.0
    amp: bool = True
    clcl_same_auc_drop_tolerance: float = 0.07
    cbcb_same_auc_drop_tolerance: float = 0.10
    cross_session_min_auc: float = 0.60
    score_range_min: float = 1e-4
    unique_score_count_min: int = 10
    practical_runtime_ms_per_pair_max: float = 100.0
    min_distribution_effect_size: float = 0.05
    reference_auc_tolerance: float = 0.03


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


def _check_descriptors(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 3 or b.ndim != 3:
        raise MatcherValidityError(f"Expected BxLxD descriptors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise MatcherValidityError(f"Incompatible descriptor batches: {tuple(a.shape)} and {tuple(b.shape)}")


def dual_softmax_scores_from_descriptors(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable confidence-weighted cosine using dual-softmax confidence."""
    _check_descriptors(a, b)
    temp = float(temperature)
    if temp <= 0.0 or not math.isfinite(temp):
        raise MatcherValidityError(f"dual-softmax temperature must be finite and positive, got {temperature}")
    sim = torch.bmm(a, b.transpose(1, 2))
    logits = sim / temp
    row_conf = F.softmax(logits, dim=2)
    col_conf = F.softmax(logits, dim=1)
    confidence = row_conf * col_conf
    mass = confidence.sum(dim=(1, 2)).clamp_min(torch.finfo(sim.dtype).eps)
    score = torch.sum(confidence * sim, dim=(1, 2)) / mass
    return score, mass


def sinkhorn_partial_assignment_scores_from_descriptors(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    entropy_regularization: float,
    iterations: int,
    dustbin_score: float,
    dustbin_mass: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable dustbin Sinkhorn score over local cosine similarities.

    Real descriptors carry total marginal mass 1/(1+dustbin_mass), and one
    extra row/column carries the predeclared unmatched mass. The returned score
    ignores dustbin transport and safely normalizes by real-real transport mass.
    """
    _check_descriptors(a, b)
    eps = float(entropy_regularization)
    if eps <= 0.0 or not math.isfinite(eps):
        raise MatcherValidityError(f"Sinkhorn entropy regularization must be positive, got {eps}")
    if int(iterations) <= 0:
        raise MatcherValidityError(f"Sinkhorn iterations must be positive, got {iterations}")
    if float(dustbin_mass) <= 0.0 or not math.isfinite(float(dustbin_mass)):
        raise MatcherValidityError(f"Sinkhorn dustbin mass must be positive, got {dustbin_mass}")

    sim = torch.bmm(a, b.transpose(1, 2))
    batch, rows, cols = sim.shape
    log_alpha = sim.new_full((batch, rows + 1, cols + 1), float(dustbin_score) / eps)
    log_alpha[:, :rows, :cols] = sim / eps

    total = 1.0 + float(dustbin_mass)
    row_real_mass = 1.0 / (float(rows) * total)
    col_real_mass = 1.0 / (float(cols) * total)
    unmatched_mass = float(dustbin_mass) / total
    row_mu = sim.new_full((batch, rows + 1), row_real_mass)
    col_nu = sim.new_full((batch, cols + 1), col_real_mass)
    row_mu[:, rows] = unmatched_mass
    col_nu[:, cols] = unmatched_mass
    log_mu = torch.log(row_mu)
    log_nu = torch.log(col_nu)

    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)
    for _ in range(int(iterations)):
        log_u = log_mu - torch.logsumexp(log_alpha + log_v[:, None, :], dim=2)
        log_v = log_nu - torch.logsumexp(log_alpha + log_u[:, :, None], dim=1)

    log_transport = log_alpha + log_u[:, :, None] + log_v[:, None, :]
    transport = torch.exp(log_transport)
    real_transport = transport[:, :rows, :cols]
    real_mass = real_transport.sum(dim=(1, 2)).clamp_min(torch.finfo(sim.dtype).eps)
    score = torch.sum(real_transport * sim, dim=(1, 2)) / real_mass
    return score, real_mass


def _detail_row(
    row: pd.Series,
    *,
    method: str,
    score: float,
    failed: bool,
    topk_a_to_b: float = float("nan"),
    topk_b_to_a: float = float("nan"),
    confidence_mass: float = float("nan"),
    transport_real_mass: float = float("nan"),
) -> dict[str, Any]:
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
        "score": float(score),
        "failed": bool(failed or not math.isfinite(float(score))),
        "topk_a_to_b": float(topk_a_to_b),
        "topk_b_to_a": float(topk_b_to_a),
        "confidence_mass": float(confidence_mass),
        "transport_real_mass": float(transport_real_mass),
    }


@torch.inference_mode()
def score_pair_frame(
    *,
    method: str,
    df: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    batch_size: int,
    cfg: MatcherAuditConfig,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    if method == S0_HARD_MNN_REFERENCE:
        scores, details, elapsed = local.score_pair_frame(
            method=local.L2_MNN,
            df=df,
            tensors=tensors,
            device=device,
            batch_size=batch_size,
            l3_shift_radius=local.PREDECLARED_SHIFT_RADIUS,
        )
        if not details.empty:
            details = details.copy()
            details["method"] = method
        return scores, details, elapsed

    if method not in DIFFERENTIABLE_SCORERS:
        raise MatcherValidityError(f"Unknown matcher: {method}")

    scores = np.full(len(df), np.nan, dtype=np.float64)
    detail_rows: list[dict[str, Any]] = []
    idx_a, idx_b, valid = local._pair_indices(df, tensors.uid_to_index)
    valid_positions = np.flatnonzero(valid)
    start_time = time.perf_counter()
    for start in range(0, len(valid_positions), int(batch_size)):
        pos = valid_positions[start : start + int(batch_size)]
        a_idx = torch.as_tensor(idx_a[pos], dtype=torch.long, device=device)
        b_idx = torch.as_tensor(idx_b[pos], dtype=torch.long, device=device)
        a = tensors.local_flat.index_select(0, a_idx)
        b = tensors.local_flat.index_select(0, b_idx)
        topk_a = np.full(len(pos), np.nan, dtype=np.float64)
        topk_b = np.full(len(pos), np.nan, dtype=np.float64)
        confidence_mass = np.full(len(pos), np.nan, dtype=np.float64)
        transport_mass = np.full(len(pos), np.nan, dtype=np.float64)

        if method == S1_CURRENT_TOPK_MAX:
            score_t, ab_t, ba_t = learned.local_pair_scores_from_descriptors(
                a,
                b,
                topk_fraction=cfg.topk_fraction,
            )
            topk_a = ab_t.detach().cpu().numpy().astype(np.float64)
            topk_b = ba_t.detach().cpu().numpy().astype(np.float64)
        elif method == S2_DUAL_SOFTMAX:
            score_t, mass_t = dual_softmax_scores_from_descriptors(
                a,
                b,
                temperature=cfg.dual_softmax_temperature,
            )
            confidence_mass = mass_t.detach().cpu().numpy().astype(np.float64)
        else:
            score_t, mass_t = sinkhorn_partial_assignment_scores_from_descriptors(
                a,
                b,
                entropy_regularization=cfg.sinkhorn_entropy_regularization,
                iterations=cfg.sinkhorn_iterations,
                dustbin_score=cfg.sinkhorn_dustbin_score,
                dustbin_mass=cfg.sinkhorn_dustbin_mass,
            )
            transport_mass = mass_t.detach().cpu().numpy().astype(np.float64)

        score_np = score_t.detach().cpu().numpy().astype(np.float64)
        scores[pos] = score_np
        for j, row_pos in enumerate(pos):
            detail_rows.append(
                _detail_row(
                    df.iloc[int(row_pos)],
                    method=method,
                    score=float(score_np[j]),
                    failed=not math.isfinite(float(score_np[j])),
                    topk_a_to_b=float(topk_a[j]),
                    topk_b_to_a=float(topk_b[j]),
                    confidence_mass=float(confidence_mass[j]),
                    transport_real_mass=float(transport_mass[j]),
                )
            )

    elapsed = time.perf_counter() - start_time
    for row_pos in np.flatnonzero(~valid):
        detail_rows.append(
            _detail_row(
                df.iloc[int(row_pos)],
                method=method,
                score=float("nan"),
                failed=True,
            )
        )
    details = pd.DataFrame(detail_rows)
    if not details.empty:
        details = details.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])
    return scores, details, float(elapsed)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _score_summary(values: np.ndarray) -> dict[str, Any]:
    finite = _finite(values)
    if finite.size == 0:
        return {
            "finite_count": 0,
            "score_mean": float("nan"),
            "score_std": float("nan"),
            "score_median": float("nan"),
            "score_min": float("nan"),
            "score_max": float("nan"),
            "score_range": float("nan"),
            "unique_score_count": 0,
            "unique_score_count_rounded_6": 0,
        }
    return {
        "finite_count": int(finite.size),
        "score_mean": float(np.mean(finite)),
        "score_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "score_median": float(np.median(finite)),
        "score_min": float(np.min(finite)),
        "score_max": float(np.max(finite)),
        "score_range": float(np.max(finite) - np.min(finite)),
        "unique_score_count": int(np.unique(finite).size),
        "unique_score_count_rounded_6": int(np.unique(np.round(finite, 6)).size),
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
    row = local.metric_row_from_scores(
        method=method,
        stage=stage,
        protocol=protocol,
        df=df,
        scores=scores,
        elapsed_seconds=elapsed_seconds,
    )
    summary = _score_summary(scores)
    row.update(
        {
            "score_min": summary["score_min"],
            "score_max": summary["score_max"],
            "score_range": summary["score_range"],
            "unique_score_count": summary["unique_score_count"],
            "unique_score_count_rounded_6": summary["unique_score_count_rounded_6"],
        }
    )
    return row


def score_distribution_rows(
    *,
    method: str,
    stage: str,
    protocol: str,
    df: pd.DataFrame,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    labels = df["label"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    groups = (
        ("all", np.ones(len(scores), dtype=bool)),
        ("genuine", labels == 1),
        ("impostor", labels == 0),
    )
    for group_name, mask in groups:
        values = scores[mask]
        summary = _score_summary(values)
        rows.append(
            {
                "method": method,
                "stage": stage,
                "protocol": protocol,
                "score_group": group_name,
                "count": int(values.size),
                "failure_count": int((~np.isfinite(values)).sum()),
                **summary,
            }
        )
    return rows


def evaluate_scorers(
    *,
    stage: str,
    pair_bundle: dict[str, pd.DataFrame],
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    cfg: MatcherAuditConfig,
    methods: Iterable[str] = ALL_SCORERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for method in methods:
        for protocol in base.CONTROL_PROTOCOLS:
            df = pair_bundle[protocol].reset_index(drop=True).copy()
            if "protocol_id" not in df.columns:
                df["protocol_id"] = protocol
            scores, _details, elapsed = score_pair_frame(
                method=method,
                df=df,
                tensors=tensors,
                device=device,
                batch_size=cfg.score_batch_size,
                cfg=cfg,
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
            distribution.extend(
                score_distribution_rows(
                    method=method,
                    stage=stage,
                    protocol=protocol,
                    df=df,
                    scores=scores,
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
    return pd.DataFrame(metric_rows), pd.DataFrame(distribution), pd.DataFrame(runtime_rows)


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


def retrieval_metrics_for_scorers(
    *,
    stage: str,
    table: pd.DataFrame,
    tensors: local.TensorDescriptorCache,
    device: torch.device,
    cfg: MatcherAuditConfig,
    methods: Iterable[str] = ALL_SCORERS,
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
            scores, _details, _elapsed = score_pair_frame(
                method=method,
                df=pd.DataFrame(pair_rows),
                tensors=tensors,
                device=device,
                batch_size=cfg.score_batch_size,
                cfg=cfg,
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


def _metric(metrics: pd.DataFrame, method: str, protocol: str) -> dict[str, Any]:
    rows = metrics[(metrics["method"] == method) & (metrics["protocol"] == protocol)]
    if rows.empty:
        raise MatcherValidityError(f"Missing metric row for {method} {protocol}")
    return dict(rows.iloc[0])


def _auc(metrics: pd.DataFrame, method: str, protocol: str) -> float:
    return float(_metric(metrics, method, protocol).get("roc_auc", float("nan")))


def _effect_size(row: dict[str, Any]) -> float:
    gap = float(row.get("genuine_score_mean", float("nan"))) - float(row.get("impostor_score_mean", float("nan")))
    gen_std = float(row.get("genuine_score_std", float("nan")))
    imp_std = float(row.get("impostor_score_std", float("nan")))
    pooled = math.sqrt(max(0.0, gen_std * gen_std + imp_std * imp_std) / 2.0)
    if not math.isfinite(gap):
        return float("nan")
    if pooled <= 1e-12:
        return float("inf") if gap > 0.0 else 0.0
    return float(gap / pooled)


def _runtime_max(runtime: pd.DataFrame, method: str) -> float:
    rows = runtime[runtime["method"] == method]
    values = pd.to_numeric(rows["runtime_ms_per_pair"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def reference_reproduction(metrics: pd.DataFrame, *, cfg: MatcherAuditConfig, smoke: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if smoke:
        return {
            "skipped": True,
            "reason": "Smoke mode changes identity and pair counts.",
            "all_within_tolerance": None,
            "rows": rows,
        }
    for method, expected_by_protocol in EXPECTED_REFERENCE_AUC.items():
        for protocol, expected in expected_by_protocol.items():
            observed = _auc(metrics, method, protocol)
            delta = observed - float(expected) if math.isfinite(observed) else float("nan")
            rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "expected_auc": float(expected),
                    "observed_auc": observed,
                    "delta": delta,
                    "tolerance": float(cfg.reference_auc_tolerance),
                    "within_tolerance": bool(math.isfinite(delta) and abs(delta) <= cfg.reference_auc_tolerance),
                }
            )
    return {
        "skipped": False,
        "all_within_tolerance": all(bool(row["within_tolerance"]) for row in rows),
        "rows": rows,
    }


def matcher_validity_decision(
    metrics: pd.DataFrame,
    runtime: pd.DataFrame,
    *,
    cfg: MatcherAuditConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    s0 = {
        "clcl_same_auc": _auc(metrics, S0_HARD_MNN_REFERENCE, CLCL_SAME),
        "cbcb_same_auc": _auc(metrics, S0_HARD_MNN_REFERENCE, CBCB_SAME),
        "clcl_cross_auc": _auc(metrics, S0_HARD_MNN_REFERENCE, CLCL_CROSS),
        "cbcb_cross_auc": _auc(metrics, S0_HARD_MNN_REFERENCE, CBCB_CROSS),
    }
    gate_rows: list[dict[str, Any]] = []
    for method in DIFFERENTIABLE_SCORERS:
        clcl_same = _auc(metrics, method, CLCL_SAME)
        cbcb_same = _auc(metrics, method, CBCB_SAME)
        clcl_cross = _auc(metrics, method, CLCL_CROSS)
        cbcb_cross = _auc(metrics, method, CBCB_CROSS)
        same_retention = bool(
            math.isfinite(clcl_same)
            and math.isfinite(cbcb_same)
            and clcl_same >= s0["clcl_same_auc"] - cfg.clcl_same_auc_drop_tolerance
            and cbcb_same >= s0["cbcb_same_auc"] - cfg.cbcb_same_auc_drop_tolerance
        )
        cross_ok = bool(
            math.isfinite(clcl_cross)
            and math.isfinite(cbcb_cross)
            and clcl_cross >= cfg.cross_session_min_auc
            and cbcb_cross >= cfg.cross_session_min_auc
        )

        finite_rows = []
        degenerate_rows = []
        distribution_rows = []
        for protocol in base.CONTROL_PROTOCOLS:
            row = _metric(metrics, method, protocol)
            pair_count = int(row.get("pair_count", 0))
            unique_min = min(int(cfg.unique_score_count_min), max(1, pair_count // 2))
            finite_rows.append(
                bool(
                    int(row.get("failure_count", row.get("failed_count", 1))) == 0
                    and int(row.get("scored_count", 0)) == pair_count
                )
            )
            degenerate_rows.append(
                bool(
                    math.isfinite(float(row.get("score_range", float("nan"))))
                    and float(row.get("score_range", 0.0)) >= cfg.score_range_min
                    and int(row.get("unique_score_count_rounded_6", 0)) >= unique_min
                )
            )
            if protocol in base.WITHIN_PROTOCOLS:
                effect = _effect_size(row)
                gap = float(row.get("genuine_score_mean", float("nan"))) - float(
                    row.get("impostor_score_mean", float("nan"))
                )
                distribution_rows.append(
                    {
                        "protocol": protocol,
                        "genuine_minus_impostor_mean": gap,
                        "standardized_effect_size": effect,
                        "passed": bool(
                            math.isfinite(gap)
                            and gap > 0.0
                            and (math.isinf(effect) or effect >= cfg.min_distribution_effect_size)
                        ),
                    }
                )

        finite_ok = all(finite_rows)
        nondegenerate_ok = all(degenerate_rows)
        max_runtime = _runtime_max(runtime, method)
        runtime_ok = bool(math.isfinite(max_runtime) and max_runtime <= cfg.practical_runtime_ms_per_pair_max)
        distribution_ok = all(bool(row["passed"]) for row in distribution_rows)
        criteria = {
            "within_same_session_auc_retention": same_retention,
            "cross_session_not_near_chance": cross_ok,
            "scores_finite": finite_ok,
            "scores_non_degenerate": nondegenerate_ok,
            "runtime_practical": runtime_ok,
            "within_modality_distributions_separated": distribution_ok,
        }
        gate_rows.append(
            {
                "method": method,
                "eligible_for_training": all(criteria.values()),
                "criteria_pass_count": int(sum(bool(v) for v in criteria.values())),
                "criteria": criteria,
                "aucs": {
                    CLCL_SAME: clcl_same,
                    CLCL_CROSS: clcl_cross,
                    CBCB_SAME: cbcb_same,
                    CBCB_CROSS: cbcb_cross,
                    CLCB: _auc(metrics, method, CLCB),
                },
                "auc_retention_vs_s0": {
                    CLCL_SAME: clcl_same - s0["clcl_same_auc"],
                    CLCL_CROSS: clcl_cross - s0["clcl_cross_auc"],
                    CBCB_SAME: cbcb_same - s0["cbcb_same_auc"],
                    CBCB_CROSS: cbcb_cross - s0["cbcb_cross_auc"],
                },
                "max_runtime_ms_per_pair": max_runtime,
                "distribution_details": distribution_rows,
            }
        )

    by_method = {row["method"]: row for row in gate_rows}
    s2_ok = bool(by_method[S2_DUAL_SOFTMAX]["eligible_for_training"])
    s3_ok = bool(by_method[S3_SINKHORN_PARTIAL_ASSIGNMENT]["eligible_for_training"])
    if s2_ok:
        classification = "A. DUAL_SOFTMAX_VALID"
        primary_reason = "S2 passed all predeclared matcher-validity criteria."
    elif s3_ok:
        classification = "B. OT_MATCHER_REQUIRED"
        primary_reason = "S2 failed, but S3 passed all predeclared matcher-validity criteria."
    else:
        partial = [
            row
            for row in (by_method[S2_DUAL_SOFTMAX], by_method[S3_SINKHORN_PARTIAL_ASSIGNMENT])
            if int(row["criteria_pass_count"]) >= 4
        ]
        if partial:
            classification = "D. MIXED_OR_INCONCLUSIVE"
            primary_reason = "No primary differentiable matcher passed every gate, but at least one result was mixed."
        else:
            classification = "C. NO_VALID_DIFFERENTIABLE_SURROGATE"
            primary_reason = "Neither S2 nor S3 preserved the hard-MNN within-modality behavior."

    return {
        "classification": classification,
        "primary_reason": primary_reason,
        "eligible_matchers": [row["method"] for row in gate_rows if row["eligible_for_training"]],
        "primary_eligible_matchers": [
            row["method"]
            for row in gate_rows
            if row["method"] in PRIMARY_DIFFERENTIABLE_SCORERS and row["eligible_for_training"]
        ],
        "s0_reference_auc": s0,
        "gate_thresholds": {
            "clcl_same_auc_drop_tolerance": cfg.clcl_same_auc_drop_tolerance,
            "cbcb_same_auc_drop_tolerance": cfg.cbcb_same_auc_drop_tolerance,
            "cross_session_min_auc": cfg.cross_session_min_auc,
            "score_range_min": cfg.score_range_min,
            "unique_score_count_min": cfg.unique_score_count_min,
            "practical_runtime_ms_per_pair_max": cfg.practical_runtime_ms_per_pair_max,
            "min_distribution_effect_size": cfg.min_distribution_effect_size,
        },
        "matcher_gate": gate_rows,
        "reference_reproduction": reference_reproduction(metrics, cfg=cfg, smoke=smoke),
        "official_val_gate": {
            "opened": False,
            "reason": "Official VAL remains closed in Phase 4B.2B.1; this is a frozen TRAIN inner-dev matcher audit.",
        },
        "test_gate": {
            "opened": False,
            "reason": "TEST is prohibited for Phase 4B.2B.1.",
        },
    }


def matcher_config_payload(cfg: MatcherAuditConfig, *, checkpoint: Path, phase4b1_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_required": DEFAULT_CHECKPOINT_SHA256,
        "phase4b_inner_split_source": str(Path(phase4b1_dir) / "inner_split.json"),
        "preprocess_contract": base.PREPROCESS_CONTRACT,
        "selected_feature_stage": local.SELECTED_STAGE_NAME,
        "selected_stage_index": int(cfg.selected_stage_index),
        "descriptor_shape": [24, 24, 256],
        "per_descriptor_l2_normalization": True,
        "scorers": {
            S0_HARD_MNN_REFERENCE: {
                "differentiable": False,
                "definition": (
                    "Exact Phase 4B.2A L2 hard mutual-nearest-neighbor score: "
                    "coverage * mean mutual-nearest-neighbor cosine; no matches score -1."
                ),
            },
            S1_CURRENT_TOPK_MAX: {
                "differentiable": True,
                "definition": (
                    "Exact Phase 4B.2B symmetric top-k local-max cosine: row/column local maxima, "
                    "mean of top ceil(topk_fraction * L) responses in each direction, then average."
                ),
                "topk_fraction": cfg.topk_fraction,
            },
            S2_DUAL_SOFTMAX: {
                "differentiable": True,
                "definition": (
                    "Cosine matrix divided by fixed temperature; row-softmax and column-softmax are multiplied "
                    "to form soft mutual confidence; score is confidence-weighted cosine normalized by confidence mass."
                ),
                "matching_temperature": cfg.dual_softmax_temperature,
            },
            S3_SINKHORN_PARTIAL_ASSIGNMENT: {
                "differentiable": True,
                "definition": (
                    "Entropy-regularized log-domain Sinkhorn transport on cosine logits with one dustbin row/column. "
                    "Score is real-real transport-weighted cosine normalized by real-real transport mass."
                ),
                "entropy_regularization": cfg.sinkhorn_entropy_regularization,
                "iterations": cfg.sinkhorn_iterations,
                "dustbin_score": cfg.sinkhorn_dustbin_score,
                "dustbin_mass": cfg.sinkhorn_dustbin_mass,
            },
        },
        "training": {
            "performed": False,
            "optimizer_instantiated": False,
            "encoder_modified": False,
        },
        "excluded_by_design": {
            "official_val": True,
            "test": True,
            "p2": True,
            "cross_attention": True,
            "ridgeformer": True,
            "broad_hyperparameter_search": True,
        },
    }


def run(
    *,
    manifest_dir: Path,
    phase4b1_dir: Path,
    checkpoint: Path,
    outdir: Path,
    polyu_root: Optional[str],
    device_arg: str,
    cfg: MatcherAuditConfig,
    smoke: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    device = local.resolve_device(device_arg)
    write_json(outdir / "matcher_config.json", matcher_config_payload(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir))

    train_images_all, resolved_root = local.load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[base.TRAIN])
    train_ids = sorted(train_images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = local.load_fixed_inner_split(train_ids, phase4b1_dir, split_seed=cfg.seed)
    inner_dev_ids = inner_split["inner_dev"]
    if smoke:
        inner_dev_ids = inner_dev_ids[: min(8, len(inner_dev_ids))]
    inner_dev_images = train_images_all[train_images_all["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()

    pair_max_pos = min(cfg.eval_max_pos, 40) if smoke else cfg.eval_max_pos
    pair_neg_per_pos = min(cfg.eval_neg_per_pos, 1) if smoke else cfg.eval_neg_per_pos
    pair_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=pair_max_pos,
        neg_per_pos=pair_neg_per_pos,
        seed=cfg.seed,
    )
    pair_counts = base.validate_pair_bundle(pair_bundle, stage="inner_dev")
    retrieval_table = base.build_retrieval_table(inner_dev_images, inner_dev_ids)

    model, ckpt_args, checkpoint_meta, checkpoint_sha = local.load_frozen_pair_model(
        checkpoint=checkpoint,
        device=device,
        require_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    input_size = int(ckpt_args.get("input_size", 384))
    image_store = local.load_image_store_for_rows(inner_dev_images, input_size=input_size)
    uids = local._unique_uids_for_pairs(pair_bundle, inner_dev_images)
    descriptor_cache = local.extract_descriptor_cache(
        encoder=model.encoder,
        image_store=image_store,
        uids=uids,
        device=device,
        batch_size=cfg.eval_batch_size,
        selected_stage_index=cfg.selected_stage_index,
        amp=cfg.amp,
    )
    tensors = local.descriptor_cache_to_tensors(descriptor_cache, device)

    metrics, distribution, runtime = evaluate_scorers(
        stage="inner_dev",
        pair_bundle=pair_bundle,
        tensors=tensors,
        device=device,
        cfg=cfg,
    )
    retrieval = retrieval_metrics_for_scorers(
        stage="inner_dev",
        table=retrieval_table,
        tensors=tensors,
        device=device,
        cfg=cfg,
    )
    decision = matcher_validity_decision(metrics, runtime, cfg=cfg, smoke=smoke)

    write_csv(outdir / "matcher_metrics.csv", metrics)
    write_csv(outdir / "retrieval_metrics.csv", retrieval)
    write_csv(outdir / "score_distribution_summary.csv", distribution)
    write_csv(outdir / "runtime_metrics.csv", runtime)
    write_json(outdir / "matcher_validity_decision.json", decision)

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
            "official_train_identity_count": int(len(train_ids)),
            "inner_dev_identity_count": int(len(inner_dev_ids)),
            "identity_disjoint": not bool(set(inner_split["inner_train"]).intersection(inner_split["inner_dev"])),
            "smoke": bool(smoke),
        },
        "pair_counts": {"inner_dev": pair_counts},
        "selected_feature_map": {
            "stage_name": descriptor_cache.selected_stage_name,
            "height": int(descriptor_cache.height),
            "width": int(descriptor_cache.width),
            "channels": int(descriptor_cache.channels),
            "descriptor_count_per_image": int(descriptor_cache.height * descriptor_cache.width),
        },
        "config": matcher_config_payload(cfg, checkpoint=checkpoint, phase4b1_dir=phase4b1_dir),
        "decision": decision,
        "outputs": {
            "matcher_config_json": str(outdir / "matcher_config.json"),
            "matcher_metrics_csv": str(outdir / "matcher_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "score_distribution_summary_csv": str(outdir / "score_distribution_summary.csv"),
            "runtime_metrics_csv": str(outdir / "runtime_metrics.csv"),
            "matcher_validity_decision_json": str(outdir / "matcher_validity_decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
        },
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
        "canonical_artifacts_not_read": {
            "pairs_val_csv": str(Path(manifest_dir) / "pairs_val.csv"),
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "Official VAL and TEST remain closed for Phase 4B.2B.1.",
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
            "frozen_encoder_only": True,
            "optimizer_instantiated": False,
            "training_performed": False,
            "encoder_modified": False,
            "official_val_read": False,
            "official_val_used_for_selection": False,
            "test_pairs_read": False,
            "test_images_loaded": False,
            "canonical_manifest_or_pairs_modified": False,
            "canonical_checkpoint_modified": False,
            "used_p2": False,
            "used_cross_attention": False,
            "implemented_ridgeformer": False,
            "broad_hyperparameter_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "metrics": metrics,
        "retrieval": retrieval,
        "distribution": distribution,
        "runtime": runtime,
        "decision": decision,
        "run_manifest": run_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.2B.1 PolyU Cross differentiable matcher validity audit.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=MatcherAuditConfig.seed)
    p.add_argument("--eval_max_pos", type=int, default=MatcherAuditConfig.eval_max_pos)
    p.add_argument("--eval_neg_per_pos", type=int, default=MatcherAuditConfig.eval_neg_per_pos)
    p.add_argument("--eval_batch_size", type=int, default=MatcherAuditConfig.eval_batch_size)
    p.add_argument("--score_batch_size", type=int, default=MatcherAuditConfig.score_batch_size)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MatcherAuditConfig(
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
            phase4b1_dir=resolve_repo_path(args.phase4b1_dir),
            checkpoint=resolve_repo_path(args.checkpoint),
            outdir=resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            device_arg=str(args.device),
            cfg=cfg,
            smoke=bool(args.smoke),
        )
    except (MatcherValidityError, local.LocalCorrespondenceError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    metrics = result["metrics"]
    retrieval = result["retrieval"]
    runtime = result["runtime"]
    print("\n=== PolyU Cross Phase 4B.2B.1 differentiable matcher validity audit complete ===")
    print(f"Output dir     : {result['outdir']}")
    print(f"Classification : {decision['classification']}")
    print(f"Eligible       : {', '.join(decision['eligible_matchers']) or 'none'}")
    print("Official VAL   : closed")
    print("TEST           : closed")
    print("\nAUC table:")
    auc_table = metrics.pivot(index="method", columns="protocol", values="roc_auc").reset_index()
    print(auc_table.to_string(index=False))
    print("\nRetrieval:")
    print(retrieval[["method", "direction", "recall_at_1", "recall_at_5", "mrr"]].to_string(index=False))
    print("\nRuntime (max ms/pair by scorer):")
    show = runtime.groupby("method", sort=False)["runtime_ms_per_pair"].max().reset_index()
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
