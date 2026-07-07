"""Phase 4B.1R: compute-adequacy replication for PolyU Cross global alignment.

This is a replication of the Phase 4B.1 global representation-alignment
experiment, not a new architecture. It keeps the same SD300 ConvEncoder
initialization, raw grayscale 384x384 / 255 preprocessing, projection dimension,
temperature, cosine scoring, inner split, and evaluation protocols. The only
intended change is a compute-adequate training regime:

* true DistributedDataParallel when launched with multiple GPUs,
* cross-rank all-gather of normalized CL/CB embeddings before InfoNCE,
* global contrastive negatives across the whole identity batch,
* seeds 13, 29, 47 for R1 and R3,
* no R2 rerun.

Official VAL is gated: it is evaluated only if the inner-dev replication reaches
COMPUTE_LIMITATION_CONFIRMED. TEST is never used for evaluation.
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
import socket
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
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.run_polyu_cross_zero_shot import git_info, safe_pkg_version, sha256_file, utc_now
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import PolyUCrossPairError

RUN_SCHEMA_VERSION = "polyu_cross_representation_alignment_compute_replication_v0"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_compute_replication_v0"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_CHECKPOINT_SHA256 = "0541c16a3e0c05638cfda2a6ccb928d0fb86988cc5f76e9524d70cfa1640584e"
REPLICATION_CONDITIONS = ("R1_projection_only", "R3_full_shared_encoder_adaptation")
REPLICATION_SEEDS = (13, 29, 47)
FINAL_CONDITION_TO_CODE = {
    "": 0,
    "R1_projection_only": 1,
    "R3_full_shared_encoder_adaptation": 3,
}
FINAL_CODE_TO_CONDITION = {code: condition for condition, code in FINAL_CONDITION_TO_CODE.items()}
ORIGINAL_INNER_DEV_AUC = {
    "R1_projection_only": 0.548,
    "R3_full_shared_encoder_adaptation": 0.553,
}


class ComputeReplicationError(RuntimeError):
    """Raised for protocol failures in Phase 4B.1R."""


@dataclass(frozen=True)
class ReplicationConfig:
    split_seed: int = 1341
    seeds: tuple[int, ...] = REPLICATION_SEEDS
    projection_dim: int = 256
    temperature: float = 0.07
    target_global_batch_identities: int = 64
    fallback_global_batch_identities: int = 32
    max_epochs: int = 30
    patience: int = 6
    projection_lr: float = 2e-4
    encoder_lr: float = 2e-5
    weight_decay: float = 1e-4
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 128
    amp: bool = True
    final_train_seed: int = 113


@dataclass(frozen=True)
class DDPContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    ddp_enabled: bool
    launched_by_torchrun: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class SeedResult:
    condition: str
    seed: int
    best_epoch: int
    best_auc: float
    best_eer: float
    best_state_dict: dict[str, torch.Tensor]
    duration_seconds: float
    trainable_param_count: int
    total_param_count: int
    trainable_names: list[str]
    metric_rows: list[dict[str, Any]]
    retrieval_rows: list[dict[str, Any]]
    collapse_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def gpu_environment() -> dict[str, Any]:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_mib": int(props.total_memory // (1024 * 1024)),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "is_kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpus": gpus,
        "torch": safe_pkg_version("torch"),
        "python": sys.version,
        "executable": sys.executable,
    }


def init_ddp_from_env() -> DDPContext:
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        device_index = local_rank % max(1, torch.cuda.device_count())
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        backend = "nccl" if os.name != "nt" else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_available():
        raise ComputeReplicationError("torch.distributed is not available")
    if not dist.is_initialized():
        if not launched:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29541")
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29541")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DDPContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
        ddp_enabled=True,
        launched_by_torchrun=launched,
    )


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def choose_global_batch(*, world_size: int, target: int, fallback: int, forced: Optional[int] = None) -> dict[str, Any]:
    if forced is not None and int(forced) > 0:
        global_batch = int(forced)
        reason = "forced_by_cli"
    elif int(world_size) >= 2:
        global_batch = int(target)
        reason = "target_global_batch_on_multi_gpu"
    else:
        global_batch = int(fallback)
        reason = "fallback_global_batch_32_world_size_lt_2"
    if global_batch % int(world_size) != 0:
        raise ComputeReplicationError(
            f"global batch identities {global_batch} must be divisible by world_size={world_size}"
        )
    local_batch = global_batch // int(world_size)
    return {
        "target_global_batch_identities": int(target),
        "fallback_global_batch_identities": int(fallback),
        "actual_global_batch_identities": int(global_batch),
        "actual_local_batch_identities": int(local_batch),
        "world_size": int(world_size),
        "fallback_reason": reason if global_batch != int(target) else "",
        "gradient_accumulation_used": False,
    }


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
        raise ComputeReplicationError("Inner split is empty")
    if set(inner["inner_train"]).intersection(inner["inner_dev"]):
        raise ComputeReplicationError("Inner split is not identity-disjoint")
    if set(inner["inner_train"]).union(inner["inner_dev"]) != set(map(str, train_ids)):
        raise ComputeReplicationError("Inner split does not match official TRAIN identities")
    return {
        "inner_train": sorted(inner["inner_train"], key=base.natural_identity_key),
        "inner_dev": sorted(inner["inner_dev"], key=base.natural_identity_key),
    }


def full_global_batches(
    pools: dict[str, dict[str, list[str]]],
    identity_ids: Iterable[str],
    *,
    global_batch_identities: int,
    epoch: int,
    seed: int,
) -> list[dict[str, Any]]:
    ids = sorted([str(x) for x in identity_ids], key=base.natural_identity_key)
    rng = np.random.default_rng(stable_int(seed, epoch, "ddp_global_epoch_order") % (2**32))
    ids = [str(x) for x in np.asarray(ids, dtype=object)[rng.permutation(len(ids))]]
    batches: list[dict[str, Any]] = []
    for start in range(0, len(ids), int(global_batch_identities)):
        chunk = ids[start : start + int(global_batch_identities)]
        if len(chunk) != int(global_batch_identities):
            continue
        cl_uids = [
            base.choose_sample_uid(pools, fu, base.CONTACTLESS, epoch=epoch, seed=seed, salt="cl")
            for fu in chunk
        ]
        cb_uids = [
            base.choose_sample_uid(pools, fu, base.CONTACT, epoch=epoch, seed=seed, salt="cb")
            for fu in chunk
        ]
        batches.append({"identity_ids": chunk, "cl_uids": cl_uids, "cb_uids": cb_uids})
    return batches


def local_slice_for_rank(batch: dict[str, Any], *, rank: int, local_batch_identities: int) -> dict[str, Any]:
    start = int(rank) * int(local_batch_identities)
    end = start + int(local_batch_identities)
    return {
        "identity_ids": batch["identity_ids"][start:end],
        "cl_uids": batch["cl_uids"][start:end],
        "cb_uids": batch["cb_uids"][start:end],
    }


def differentiable_all_gather(tensor: torch.Tensor, *, world_size: int) -> torch.Tensor:
    if int(world_size) == 1:
        return tensor
    try:
        from torch.distributed.nn.functional import all_gather

        gathered = all_gather(tensor)
        return torch.cat(list(gathered), dim=0)
    except Exception as exc:  # pragma: no cover - only when DDP backend lacks autograd gather
        raise ComputeReplicationError("Differentiable torch.distributed.nn.functional.all_gather failed") from exc


def contrastive_loss_with_global_negatives(
    z_cl_local: torch.Tensor,
    z_cb_local: torch.Tensor,
    z_cl_all: torch.Tensor,
    z_cb_all: torch.Tensor,
    *,
    rank: int,
    temperature: float,
) -> torch.Tensor:
    local_n = int(z_cl_local.shape[0])
    target = torch.arange(local_n, device=z_cl_local.device) + int(rank) * local_n
    logits_cl_to_cb = z_cl_local @ z_cb_all.T / float(temperature)
    logits_cb_to_cl = z_cb_local @ z_cl_all.T / float(temperature)
    return 0.5 * (
        F.cross_entropy(logits_cl_to_cb, target)
        + F.cross_entropy(logits_cb_to_cl, target)
    )


def ddp_symmetric_infonce(
    z_cl_local: torch.Tensor,
    z_cb_local: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    temperature: float,
) -> torch.Tensor:
    z_cl_all = differentiable_all_gather(z_cl_local, world_size=world_size)
    z_cb_all = differentiable_all_gather(z_cb_local, world_size=world_size)
    return contrastive_loss_with_global_negatives(
        z_cl_local,
        z_cb_local,
        z_cl_all,
        z_cb_all,
        rank=rank,
        temperature=temperature,
    )


def average_float_across_ranks(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(world_size)
    return float(tensor.item())


def broadcast_float_from_rank0(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if int(world_size) > 1:
        dist.broadcast(tensor, src=0)
    return float(tensor.item())


def encode_final_gate(
    *,
    opened: bool,
    selected_condition: str,
    selected_epoch: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode the rank-0 final-train decision into a small broadcast tensor."""
    condition = str(selected_condition)
    epoch = int(selected_epoch)
    if bool(opened):
        if condition not in REPLICATION_CONDITIONS:
            raise ComputeReplicationError(
                f"Opened final gate requires a replication condition; got {condition!r}"
            )
        if epoch < 1:
            raise ComputeReplicationError(
                f"Opened final gate requires selected_epoch >= 1; got {epoch}"
            )
    else:
        condition = ""
        epoch = 0
    return torch.tensor(
        [
            int(bool(opened)),
            int(FINAL_CONDITION_TO_CODE[condition]),
            int(epoch),
        ],
        dtype=torch.int64,
        device=device,
    )


def decode_final_gate(gate: torch.Tensor) -> dict[str, Any]:
    """Decode and validate the broadcast final-train decision tensor."""
    values = gate.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    if len(values) != 3:
        raise ComputeReplicationError(f"Final gate tensor must contain 3 integers; got {values}")
    opened_raw, condition_code, selected_epoch = (int(x) for x in values)
    if opened_raw not in (0, 1):
        raise ComputeReplicationError(f"Final gate opened flag must be 0 or 1; got {opened_raw}")
    if condition_code not in FINAL_CODE_TO_CONDITION:
        raise ComputeReplicationError(f"Unknown final condition code: {condition_code}")
    opened = bool(opened_raw)
    selected_condition = FINAL_CODE_TO_CONDITION[condition_code]
    if opened:
        if selected_condition not in REPLICATION_CONDITIONS:
            raise ComputeReplicationError(
                f"Opened final gate decoded without a valid condition: code={condition_code}"
            )
        if selected_epoch < 1:
            raise ComputeReplicationError(
                f"Opened final gate requires selected_epoch >= 1; got {selected_epoch}"
            )
    else:
        if selected_condition or selected_epoch != 0:
            raise ComputeReplicationError(
                "Closed final gate must carry empty condition and selected_epoch=0"
            )
    return {
        "opened": opened,
        "selected_condition": selected_condition,
        "selected_epoch": int(selected_epoch),
    }


def broadcast_final_gate(
    ctx: DDPContext,
    *,
    opened: bool = False,
    selected_condition: str = "",
    selected_epoch: int = 0,
) -> dict[str, Any]:
    """Broadcast rank 0's A-gate decision so every rank follows the same branch."""
    if ctx.is_main:
        gate = encode_final_gate(
            opened=opened,
            selected_condition=selected_condition,
            selected_epoch=selected_epoch,
            device=ctx.device,
        )
    else:
        gate = torch.zeros(3, dtype=torch.int64, device=ctx.device)
    dist.broadcast(gate, src=0)
    return decode_final_gate(gate)


def wrap_ddp(model: nn.Module, ctx: DDPContext) -> DistributedDataParallel:
    kwargs: dict[str, Any] = {"find_unused_parameters": False}
    if ctx.device.type == "cuda" and ctx.backend == "nccl":
        kwargs["device_ids"] = [ctx.device.index]
        kwargs["output_device"] = ctx.device.index
    return DistributedDataParallel(model, **kwargs)


def disable_inplace_activations(module: nn.Module) -> None:
    """Keep the checkpoint math but avoid DDP/all-gather autograd versioning
    conflicts through in-place SiLU/ReLU activations during R3 training."""
    for child in module.modules():
        if hasattr(child, "inplace"):
            try:
                child.inplace = False
            except Exception:
                pass


def train_ddp_condition(
    *,
    condition: str,
    seed: int,
    ctx: DDPContext,
    cfg: ReplicationConfig,
    global_batch_info: dict[str, Any],
    checkpoint: Path,
    inner_train_ids: list[str],
    inner_dev_pair_bundle: dict[str, pd.DataFrame],
    inner_dev_retrieval: pd.DataFrame,
    inner_train_images: pd.DataFrame,
    inner_dev_images: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    fixed_epochs: Optional[int] = None,
    eval_each_epoch: bool = True,
) -> Optional[SeedResult]:
    seed_everything(int(seed) + ctx.rank * 100_003)
    train_cfg = base.TrainConfig(
        seed=int(seed),
        projection_dim=cfg.projection_dim,
        temperature=cfg.temperature,
        batch_identities=int(global_batch_info["actual_global_batch_identities"]),
        max_epochs=int(fixed_epochs or cfg.max_epochs),
        patience=int(cfg.patience),
        projection_lr=cfg.projection_lr,
        encoder_lr=cfg.encoder_lr,
        weight_decay=cfg.weight_decay,
        eval_max_pos=cfg.eval_max_pos,
        eval_neg_per_pos=cfg.eval_neg_per_pos,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
    )
    bundle = base.build_model_bundle(
        checkpoint=checkpoint,
        condition=condition,
        projection_dim=cfg.projection_dim,
        device=ctx.device,
        seed=int(seed),
    )
    disable_inplace_activations(bundle.model)
    ddp_model = wrap_ddp(bundle.model, ctx)
    module = ddp_model.module
    projection_params = [p for p in module.projection.parameters() if p.requires_grad]
    encoder_params = [p for name, p in module.named_parameters() if name.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if projection_params:
        param_groups.append({"params": projection_params, "lr": cfg.projection_lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.encoder_lr})
    if not param_groups:
        raise ComputeReplicationError(f"{condition} has no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and ctx.device.type == "cuda")
    pools = base.build_identity_pools(inner_train_images, inner_train_ids)

    best_auc = -math.inf
    best_epoch = 0
    best_eer = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: list[dict[str, Any]] = []
    best_retrieval: list[dict[str, Any]] = []
    best_collapse: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    no_improve = 0
    start_time = time.perf_counter()

    for epoch in range(1, int(fixed_epochs or cfg.max_epochs) + 1):
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()
        base.set_training_modes(module, condition)
        ddp_model.train()
        losses: list[float] = []
        global_batches = full_global_batches(
            pools,
            inner_train_ids,
            global_batch_identities=int(global_batch_info["actual_global_batch_identities"]),
            epoch=epoch,
            seed=int(seed),
        )
        for global_batch in global_batches:
            local_batch = local_slice_for_rank(
                global_batch,
                rank=ctx.rank,
                local_batch_identities=int(global_batch_info["actual_local_batch_identities"]),
            )
            if len(local_batch["identity_ids"]) != int(global_batch_info["actual_local_batch_identities"]):
                continue
            cl = base.stack_batch(image_store, local_batch["cl_uids"], device=ctx.device)
            cb = base.stack_batch(image_store, local_batch["cb_uids"], device=ctx.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and ctx.device.type == "cuda"):
                z = ddp_model(torch.cat([cl, cb], dim=0))
                z_cl, z_cb = z.chunk(2, dim=0)
                loss = ddp_symmetric_infonce(
                    z_cl,
                    z_cb,
                    rank=ctx.rank,
                    world_size=ctx.world_size,
                    temperature=cfg.temperature,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        train_loss = average_float_across_ranks(float(np.mean(losses)) if losses else float("nan"), ctx.device, ctx.world_size)

        metrics: list[dict[str, Any]] = []
        retrieval: list[dict[str, Any]] = []
        collapse: list[dict[str, Any]] = []
        clcb_auc = float("nan")
        clcb_eer = float("nan")
        if eval_each_epoch and ctx.is_main:
            metrics, retrieval, collapse = base.evaluate_condition(
                model=module,
                condition=condition,
                stage="inner_dev",
                pair_bundle=inner_dev_pair_bundle,
                retrieval_table=inner_dev_retrieval,
                eval_images=inner_dev_images,
                image_store=image_store,
                device=ctx.device,
                eval_batch_size=cfg.eval_batch_size,
                amp=cfg.amp,
                epoch=epoch,
            )
            clcb = next(r for r in metrics if r["protocol"] == "contactless_to_contact_based")
            clcb_auc = float(clcb["roc_auc"])
            clcb_eer = float(clcb["eer"])
        clcb_auc = broadcast_float_from_rank0(clcb_auc, ctx.device, ctx.world_size)
        clcb_eer = broadcast_float_from_rank0(clcb_eer, ctx.device, ctx.world_size)

        if ctx.is_main:
            curve_rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "inner_dev_clcb_auc": clcb_auc if eval_each_epoch else "",
                    "inner_dev_eer": clcb_eer if eval_each_epoch else "",
                    "seconds_elapsed": float(time.perf_counter() - start_time),
                    "global_batch_identities": int(global_batch_info["actual_global_batch_identities"]),
                    "local_batch_identities": int(global_batch_info["actual_local_batch_identities"]),
                }
            )
            print(json.dumps(curve_rows[-1], ensure_ascii=True), flush=True)

        if eval_each_epoch:
            if math.isfinite(clcb_auc) and clcb_auc > best_auc + 1e-6:
                best_auc = clcb_auc
                best_eer = clcb_eer
                best_epoch = int(epoch)
                no_improve = 0
                if ctx.is_main:
                    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                    best_metrics = copy.deepcopy(metrics)
                    best_retrieval = copy.deepcopy(retrieval)
                    best_collapse = copy.deepcopy(collapse)
            else:
                no_improve += 1
            stop = no_improve >= int(cfg.patience)
            stop_value = broadcast_float_from_rank0(1.0 if stop else 0.0, ctx.device, ctx.world_size)
            if bool(stop_value):
                break

    if not eval_each_epoch:
        if ctx.is_main:
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            best_epoch = int(fixed_epochs or cfg.max_epochs)
            best_auc = float("nan")
        else:
            return None

    if not ctx.is_main:
        return None
    if best_state is None:
        raise ComputeReplicationError(f"{condition}/seed={seed} did not produce a best state")
    return SeedResult(
        condition=condition,
        seed=int(seed),
        best_epoch=int(best_epoch),
        best_auc=float(best_auc),
        best_eer=float(best_eer),
        best_state_dict=best_state,
        duration_seconds=float(time.perf_counter() - start_time),
        trainable_param_count=bundle.trainable_param_count,
        total_param_count=bundle.total_param_count,
        trainable_names=bundle.trainable_names,
        metric_rows=best_metrics,
        retrieval_rows=best_retrieval,
        collapse_rows=best_collapse,
        curve_rows=curve_rows,
    )


def flatten_seed_result(result: SeedResult) -> dict[str, Any]:
    clcb = next(r for r in result.metric_rows if r["protocol"] == "contactless_to_contact_based")
    retrieval = {r["direction"]: r for r in result.retrieval_rows}
    within = {r["protocol"]: r for r in result.metric_rows if r["protocol"] != "contactless_to_contact_based"}
    return {
        "condition": result.condition,
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "duration_seconds": result.duration_seconds,
        "trainable_param_count": result.trainable_param_count,
        "total_param_count": result.total_param_count,
        "inner_dev_clcb_auc": float(clcb["roc_auc"]),
        "inner_dev_clcb_eer": float(clcb["eer"]),
        "cl_to_cb_recall_at_1": retrieval.get("CL_probe_to_CB_gallery", {}).get("recall_at_1", float("nan")),
        "cl_to_cb_recall_at_5": retrieval.get("CL_probe_to_CB_gallery", {}).get("recall_at_5", float("nan")),
        "cl_to_cb_mrr": retrieval.get("CL_probe_to_CB_gallery", {}).get("mrr", float("nan")),
        "cb_to_cl_recall_at_1": retrieval.get("CB_probe_to_CL_gallery", {}).get("recall_at_1", float("nan")),
        "cb_to_cl_recall_at_5": retrieval.get("CB_probe_to_CL_gallery", {}).get("recall_at_5", float("nan")),
        "cb_to_cl_mrr": retrieval.get("CB_probe_to_CL_gallery", {}).get("mrr", float("nan")),
        "mean_cross_modal_mrr": float(
            np.nanmean(
                [
                    retrieval.get("CL_probe_to_CB_gallery", {}).get("mrr", float("nan")),
                    retrieval.get("CB_probe_to_CL_gallery", {}).get("mrr", float("nan")),
                ]
            )
        ),
        "clcl_same_auc": within.get("contactless_to_contactless_same_session", {}).get("roc_auc", float("nan")),
        "clcl_cross_auc": within.get("contactless_to_contactless_cross_session", {}).get("roc_auc", float("nan")),
        "cbcb_same_auc": within.get("contact_based_to_contact_based_same_session", {}).get("roc_auc", float("nan")),
        "cbcb_cross_auc": within.get("contact_based_to_contact_based_cross_session", {}).get("roc_auc", float("nan")),
        "within_mean_auc": float(np.nanmean([float(r["roc_auc"]) for r in within.values()])),
    }


def aggregate_seed_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "inner_dev_clcb_auc",
        "inner_dev_clcb_eer",
        "mean_cross_modal_mrr",
        "cl_to_cb_recall_at_1",
        "cl_to_cb_recall_at_5",
        "cl_to_cb_mrr",
        "cb_to_cl_recall_at_1",
        "cb_to_cl_recall_at_5",
        "cb_to_cl_mrr",
        "clcl_same_auc",
        "clcl_cross_auc",
        "cbcb_same_auc",
        "cbcb_cross_auc",
        "within_mean_auc",
        "best_epoch",
        "duration_seconds",
    ]
    for condition, group in seed_metrics.groupby("condition", sort=False):
        row: dict[str, Any] = {"condition": condition, "seed_count": int(len(group))}
        original = ORIGINAL_INNER_DEV_AUC.get(str(condition), float("nan"))
        row["original_inner_dev_clcb_auc"] = original
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            row[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_min"] = float(np.min(values)) if values.size else float("nan")
            row[f"{metric}_max"] = float(np.max(values)) if values.size else float("nan")
        auc_values = pd.to_numeric(group["inner_dev_clcb_auc"], errors="coerce").to_numpy(dtype=float)
        row["median_auc_minus_original"] = float(np.nanmedian(auc_values) - original)
        row["seeds_improved_by_at_least_0.05"] = int(np.sum((auc_values - original) >= 0.05))
        rows.append(row)
    return pd.DataFrame(rows)


def original_phase4b1_reference(phase4b1_dir: Path) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "inner_dev_auc": dict(ORIGINAL_INNER_DEV_AUC),
        "inner_dev_mean_cross_modal_mrr": {},
        "inner_dev_within_mean_auc": {},
    }
    metrics_path = Path(phase4b1_dir) / "inner_dev_metrics.csv"
    retrieval_path = Path(phase4b1_dir) / "retrieval_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for condition in REPLICATION_CONDITIONS:
            rows = metrics[(metrics["condition"] == condition) & (metrics["protocol"] != "contactless_to_contact_based")]
            if not rows.empty:
                reference["inner_dev_within_mean_auc"][condition] = float(pd.to_numeric(rows["roc_auc"], errors="coerce").mean())
    if retrieval_path.exists():
        retrieval = pd.read_csv(retrieval_path)
        for condition in REPLICATION_CONDITIONS:
            rows = retrieval[(retrieval["condition"] == condition) & (retrieval["stage"] == "inner_dev")]
            if not rows.empty:
                reference["inner_dev_mean_cross_modal_mrr"][condition] = float(pd.to_numeric(rows["mrr"], errors="coerce").mean())
    return reference


def classify_compute_adequacy(
    seed_metrics: pd.DataFrame,
    seed_summary: pd.DataFrame,
    collapse_rows: pd.DataFrame,
    *,
    reference: dict[str, Any],
) -> dict[str, Any]:
    condition_details: list[dict[str, Any]] = []
    any_A = False
    all_B = True
    any_large_single = False
    for _, summary in seed_summary.iterrows():
        condition = str(summary["condition"])
        original_auc = float(reference["inner_dev_auc"].get(condition, ORIGINAL_INNER_DEV_AUC[condition]))
        median_auc = float(summary["inner_dev_clcb_auc_median"])
        median_gain = median_auc - original_auc
        improved_seed_count = int(summary["seeds_improved_by_at_least_0.05"])
        rows = seed_metrics[seed_metrics["condition"] == condition].copy()
        original_mrr = float(reference.get("inner_dev_mean_cross_modal_mrr", {}).get(condition, float("nan")))
        if math.isfinite(original_mrr):
            retrieval_improved = int(np.sum(pd.to_numeric(rows["mean_cross_modal_mrr"], errors="coerce") >= original_mrr + 0.03))
            retrieval_flat = abs(float(summary["mean_cross_modal_mrr_median"]) - original_mrr) < 0.02
        else:
            retrieval_improved = 0
            retrieval_flat = True
        original_within = float(reference.get("inner_dev_within_mean_auc", {}).get(condition, float("nan")))
        within_ok = True
        if math.isfinite(original_within):
            within_ok = float(summary["within_mean_auc_median"]) >= original_within - 0.05
        crows = collapse_rows[collapse_rows["condition"] == condition]
        no_collapse = True
        if not crows.empty:
            no_collapse = bool(
                (pd.to_numeric(crows["per_dim_std_mean"], errors="coerce") > 1e-3).all()
                and (pd.to_numeric(crows["near_identical_fraction"], errors="coerce") < 0.05).all()
            )
        stable_A = median_gain >= 0.05 and improved_seed_count >= 2 and retrieval_improved >= 2 and no_collapse and within_ok
        condition_B = median_gain < 0.02 and retrieval_flat
        any_A = any_A or stable_A
        all_B = all_B and condition_B
        any_large_single = any_large_single or bool((pd.to_numeric(rows["inner_dev_clcb_auc"], errors="coerce") - original_auc >= 0.05).any())
        condition_details.append(
            {
                "condition": condition,
                "original_inner_dev_clcb_auc": original_auc,
                "median_inner_dev_clcb_auc": median_auc,
                "median_auc_minus_original": median_gain,
                "seeds_improved_by_at_least_0.05": improved_seed_count,
                "retrieval_seeds_improved_by_at_least_0.03_mrr": retrieval_improved,
                "retrieval_flat": bool(retrieval_flat),
                "within_modality_ok": bool(within_ok),
                "no_embedding_collapse": bool(no_collapse),
                "meets_A": bool(stable_A),
                "meets_B_for_condition": bool(condition_B),
            }
        )
    if any_A:
        label = "A. COMPUTE_LIMITATION_CONFIRMED"
    elif all_B and not any_large_single:
        label = "B. GLOBAL_ALIGNMENT_FAILURE_CONFIRMED"
    else:
        label = "C. OPTIMIZATION_INSTABILITY"
    return {
        "classification": label,
        "condition_details": condition_details,
        "official_val_gate": {
            "opened": bool(label.startswith("A.")),
            "reason": (
                "A reached: compute-adequate inner-dev gains are stable."
                if label.startswith("A.")
                else "A not reached: official VAL remains closed for this replication."
            ),
        },
        "has_isolated_high_seed": bool(any_large_single),
    }


def choose_final_condition_for_A(seed_summary: pd.DataFrame) -> tuple[str, int]:
    best_median = float(seed_summary["inner_dev_clcb_auc_median"].max())
    eligible = seed_summary[seed_summary["inner_dev_clcb_auc_median"] >= best_median - 0.02].copy()
    rank = {"R1_projection_only": 1, "R3_full_shared_encoder_adaptation": 3}
    eligible["_rank"] = eligible["condition"].map(rank)
    selected = eligible.sort_values(["_rank", "condition"], kind="mergesort").iloc[0]
    epoch = int(round(float(selected["best_epoch_median"])))
    return str(selected["condition"]), max(1, epoch)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(path, index=False)


def run_worker(args: argparse.Namespace) -> None:
    ctx = init_ddp_from_env()
    outdir = resolve_repo_path(args.outdir)
    manifest_dir = resolve_repo_path(args.data_dir)
    controls_dir = resolve_repo_path(args.controls_dir)
    checkpoint = resolve_repo_path(args.checkpoint)
    phase4b1_dir = resolve_repo_path(args.phase4b1_dir)
    cfg = ReplicationConfig(
        seeds=tuple(int(x) for x in str(args.seeds).split(",") if x.strip()),
        target_global_batch_identities=int(args.target_global_batch_identities),
        fallback_global_batch_identities=int(args.fallback_global_batch_identities),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        amp=not bool(args.no_amp),
    )
    forced_batch = int(args.global_batch_identities) if int(args.global_batch_identities) > 0 else None
    global_batch_info = choose_global_batch(
        world_size=ctx.world_size,
        target=cfg.target_global_batch_identities,
        fallback=cfg.fallback_global_batch_identities,
        forced=forced_batch,
    )

    if ctx.is_main:
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(outdir / "kaggle_environment.json", {**gpu_environment(), "ddp": ctx.__dict__})
        write_json(
            outdir / "global_batch_validation.json",
            {
                **global_batch_info,
                "info_nce_global_negatives": True,
                "cross_rank_embedding_all_gather": ctx.world_size > 1,
                "differentiable_all_gather_function": "torch.distributed.nn.functional.all_gather",
                "loss_scope": "local queries against gathered global CL/CB keys",
                "negatives_per_query": int(global_batch_info["actual_global_batch_identities"]) - 1,
                "no_gradient_accumulation_substitute": True,
            },
        )

    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != DEFAULT_CHECKPOINT_SHA256:
        raise ComputeReplicationError(f"Checkpoint SHA mismatch: {checkpoint_sha} != {DEFAULT_CHECKPOINT_SHA256}")

    images, resolved_root = base.load_train_val_manifest(manifest_dir, polyu_root=str(args.polyu_root).strip() or None)
    train_images = images[images["split"] == base.TRAIN].copy()
    train_ids = sorted(train_images["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
    inner_split = load_fixed_inner_split(train_ids, phase4b1_dir, split_seed=cfg.split_seed)
    inner_train_ids = inner_split["inner_train"]
    inner_dev_ids = inner_split["inner_dev"]
    inner_train_images = train_images[train_images["finger_unit_id"].astype(str).isin(set(inner_train_ids))].copy()
    inner_dev_images = train_images[train_images["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    inner_bundle = base.build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=cfg.eval_max_pos,
        neg_per_pos=cfg.eval_neg_per_pos,
        seed=cfg.split_seed,
    )
    base.validate_pair_bundle(inner_bundle, stage="inner_dev")
    inner_retrieval = base.build_retrieval_table(inner_dev_images, inner_dev_ids)
    image_store = base.load_image_store(images, input_size=384)

    if ctx.is_main:
        write_json(
            outdir / "experiment_config.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "phase4b1_reference_dir": str(phase4b1_dir),
                "conditions": list(REPLICATION_CONDITIONS),
                "seeds": list(cfg.seeds),
                "projection_dim": cfg.projection_dim,
                "temperature": cfg.temperature,
                "max_epochs": cfg.max_epochs,
                "patience": cfg.patience,
                "optimizer": {
                    "family": "AdamW",
                    "projection_lr": cfg.projection_lr,
                    "encoder_lr": cfg.encoder_lr,
                    "weight_decay": cfg.weight_decay,
                    "broad_hyperparameter_search": False,
                },
                "preprocess_contract": base.PREPROCESS_CONTRACT,
                "augmentation_policy": "none",
                "inner_split_source": str(phase4b1_dir / "inner_split.json"),
                "official_val_gate": "VAL only opened if COMPUTE_LIMITATION_CONFIRMED",
                "original_inner_dev_auc": ORIGINAL_INNER_DEV_AUC,
            },
        )

    seed_results: list[SeedResult] = []
    for condition in REPLICATION_CONDITIONS:
        for seed in cfg.seeds:
            if ctx.is_main:
                print(f"[train-ddp] condition={condition} seed={seed}", flush=True)
            result = train_ddp_condition(
                condition=condition,
                seed=int(seed),
                ctx=ctx,
                cfg=cfg,
                global_batch_info=global_batch_info,
                checkpoint=checkpoint,
                inner_train_ids=inner_train_ids,
                inner_dev_pair_bundle=inner_bundle,
                inner_dev_retrieval=inner_retrieval,
                inner_train_images=inner_train_images,
                inner_dev_images=inner_dev_images,
                image_store=image_store,
            )
            if result is not None:
                seed_results.append(result)
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()

    decision: Optional[dict[str, Any]] = None
    seed_metrics = pd.DataFrame()
    seed_summary = pd.DataFrame()
    retrieval_df = pd.DataFrame()
    within_df = pd.DataFrame()
    collapse_df = pd.DataFrame()
    selected_condition = ""
    selected_epoch = 0

    if ctx.is_main:
        seed_metric_rows = [flatten_seed_result(r) for r in seed_results]
        training_curve_rows = [row for r in seed_results for row in r.curve_rows]
        retrieval_rows = []
        within_rows = []
        collapse_rows = []
        for r in seed_results:
            for row in r.retrieval_rows:
                retrieval_rows.append({**row, "seed": r.seed})
            for row in r.metric_rows:
                if row["protocol"] != "contactless_to_contact_based":
                    within_rows.append({**row, "seed": r.seed})
            for row in r.collapse_rows:
                collapse_rows.append({**row, "seed": r.seed})

        seed_metrics = pd.DataFrame(seed_metric_rows)
        seed_summary = aggregate_seed_summary(seed_metrics)
        retrieval_df = pd.DataFrame(retrieval_rows)
        within_df = pd.DataFrame(within_rows)
        collapse_df = pd.DataFrame(collapse_rows)
        reference = original_phase4b1_reference(phase4b1_dir)
        decision = classify_compute_adequacy(seed_metrics, seed_summary, collapse_df, reference=reference)

        write_csv(outdir / "seed_metrics.csv", seed_metrics)
        write_csv(outdir / "seed_summary.csv", seed_summary)
        write_csv(outdir / "training_curves.csv", training_curve_rows)
        write_csv(outdir / "retrieval_metrics.csv", retrieval_df)
        write_csv(outdir / "within_modality_controls.csv", within_df)
        write_csv(outdir / "embedding_collapse_diagnostics.csv", collapse_df)

        if decision["official_val_gate"]["opened"]:
            selected_condition, selected_epoch = choose_final_condition_for_A(seed_summary)
            decision["official_val_gate"]["selected_condition"] = selected_condition
            decision["official_val_gate"][
                "selected_epoch_policy"
            ] = "rounded median best epoch across selected condition seeds"
            decision["official_val_gate"]["selected_epoch"] = selected_epoch

    final_gate = broadcast_final_gate(
        ctx,
        opened=(bool(decision["official_val_gate"]["opened"]) if ctx.is_main and decision is not None else False),
        selected_condition=selected_condition,
        selected_epoch=selected_epoch,
    )

    final: Optional[SeedResult] = None
    if final_gate["opened"]:
        final = train_ddp_condition(
            condition=str(final_gate["selected_condition"]),
            seed=cfg.final_train_seed,
            ctx=ctx,
            cfg=cfg,
            global_batch_info=global_batch_info,
            checkpoint=checkpoint,
            inner_train_ids=train_ids,
            inner_dev_pair_bundle=inner_bundle,
            inner_dev_retrieval=inner_retrieval,
            inner_train_images=train_images,
            inner_dev_images=inner_dev_images,
            image_store=image_store,
            fixed_epochs=int(final_gate["selected_epoch"]),
            eval_each_epoch=False,
        )

    # No distributed collectives are needed after the optional final DDP train.
    # Tear down the process group on all ranks together before rank 0 performs
    # the potentially long official-VAL evaluation.
    cleanup_ddp()
    if not ctx.is_main:
        return

    if decision is None:
        raise ComputeReplicationError("Rank 0 did not produce a compute-adequacy decision")

    final_val_rows: list[dict[str, Any]] = []
    if final_gate["opened"]:
        if final is None:
            raise ComputeReplicationError("Rank 0 final DDP training did not return a final state")
        official_val_bundle = base.load_official_val_pair_bundle(manifest_dir, controls_dir)
        val_images = images[images["split"] == base.VAL].copy()
        val_ids = sorted(
            val_images["finger_unit_id"].astype(str).unique().tolist(),
            key=base.natural_identity_key,
        )
        val_retrieval = base.build_retrieval_table(val_images, val_ids)
        final_bundle = base.build_model_bundle(
            checkpoint=checkpoint,
            condition=str(final_gate["selected_condition"]),
            projection_dim=cfg.projection_dim,
            device=ctx.device,
            seed=cfg.final_train_seed,
        )
        final_bundle.model.load_state_dict(final.best_state_dict, strict=True)
        final_val_rows, final_ret, final_collapse = base.evaluate_condition(
            model=final_bundle.model,
            condition=f"compute_adequate_final_{final_gate['selected_condition']}",
            stage="official_val",
            pair_bundle=official_val_bundle,
            retrieval_table=val_retrieval,
            eval_images=val_images,
            image_store=image_store,
            device=ctx.device,
            eval_batch_size=cfg.eval_batch_size,
            amp=cfg.amp,
            epoch=int(final_gate["selected_epoch"]),
        )
        write_csv(outdir / "final_val_metrics.csv", final_val_rows)
        write_csv(outdir / "final_val_retrieval_metrics.csv", final_ret)
        write_csv(outdir / "final_val_embedding_collapse_diagnostics.csv", final_collapse)
    else:
        decision["official_val_gate"]["retained_phase4b1_official_val"] = str(
            phase4b1_dir / "final_val_metrics.csv"
        )

    write_json(outdir / "compute_adequacy_decision.json", decision)

    canonical_files = {
        "manifest_csv": manifest_dir / "manifest.csv",
        "pairs_train_csv": manifest_dir / "pairs_train.csv",
        "pairs_val_csv": manifest_dir / "pairs_val.csv",
        "pairs_test_csv_integrity_only": manifest_dir / "pairs_test.csv",
        "checkpoint": checkpoint,
    }
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "dataset": base.DATASET_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "ddp": ctx.__dict__,
        "global_batch": global_batch_info,
        "identity_counts": {
            "official_train": len(train_ids),
            "inner_train": len(inner_train_ids),
            "inner_dev": len(inner_dev_ids),
        },
        "outputs": {
            "kaggle_environment_json": str(outdir / "kaggle_environment.json"),
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "global_batch_validation_json": str(outdir / "global_batch_validation.json"),
            "seed_metrics_csv": str(outdir / "seed_metrics.csv"),
            "seed_summary_csv": str(outdir / "seed_summary.csv"),
            "training_curves_csv": str(outdir / "training_curves.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "within_modality_controls_csv": str(outdir / "within_modality_controls.csv"),
            "embedding_collapse_diagnostics_csv": str(outdir / "embedding_collapse_diagnostics.csv"),
            "compute_adequacy_decision_json": str(outdir / "compute_adequacy_decision.json"),
            "final_val_metrics_csv": str(outdir / "final_val_metrics.csv") if final_val_rows else "",
        },
        "decision": decision,
        "canonical_artifact_sha256": {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()},
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
            "test_pairs_scored": False,
            "test_images_loaded": False,
            "official_val_used_for_early_stopping": False,
            "official_val_opened_under_gate": bool(decision["official_val_gate"]["opened"]),
            "modified_manifest_or_pairs": False,
            "modified_checkpoint": False,
            "changed_preprocessing": False,
            "used_p2_preprocessing": False,
            "introduced_augmentation": False,
            "ran_R2": False,
            "used_gradient_accumulation_as_negative_substitute": False,
            "used_fusion_scores": False,
            "used_local_attention_or_transformer": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    print("\n=== PolyU Cross 4B.1R compute replication complete ===")
    print(f"Output dir     : {outdir}")
    print(f"Classification : {decision['classification']}")
    print(seed_metrics[["condition", "seed", "best_epoch", "inner_dev_clcb_auc", "mean_cross_modal_mrr"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.1R compute-adequacy DDP replication.")
    p.add_argument("--data_dir", type=str, default=base.DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=base.DEFAULT_CONTROLS_DIR)
    p.add_argument("--checkpoint", type=str, default=base.DEFAULT_CHECKPOINT)
    p.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--seeds", type=str, default="13,29,47")
    p.add_argument("--target_global_batch_identities", type=int, default=64)
    p.add_argument("--fallback_global_batch_identities", type=int, default=32)
    p.add_argument("--global_batch_identities", type=int, default=0, help="Force actual global batch; 0=auto.")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--no_amp", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_worker(args)
    except (ComputeReplicationError, base.AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        try:
            cleanup_ddp()
        except Exception:
            pass
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
