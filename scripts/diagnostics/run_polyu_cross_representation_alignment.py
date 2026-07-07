"""Minimal cross-modal representation alignment for PolyU Cross (Phase 4B.1).

This diagnostic answers a narrow capacity question: how much adaptation is
needed to make the existing SD300 deep encoder produce a shared CL/CB embedding
space on PolyU Cross?

Protocol guardrails:
* TRAIN identities only for fitting.
* One deterministic identity-disjoint inner split from TRAIN for early stopping
  and R1/R2/R3 selection.
* Official VAL is evaluated once after the selected configuration is frozen.
* TEST pairs/images are never scored.
* The canonical manifest, pair bundles, and SD300 checkpoint are read-only.
* No P2 preprocessing, SourceAFIS/SIFT modification, fusion target, classifier
  softmax objective, separate modality encoders, transformer, GAN, or unwarping.

The primary training objective is symmetric cross-modal InfoNCE over identity
balanced batches. Each batch contains N finger_unit_ids, one contactless image
and one contact-based image per identity; all other identities in the batch are
negatives. Embeddings are L2-normalized and pairs are scored by cosine.
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

from pipelines.benchmark.run_polyu_cross_zero_shot import (
    git_info,
    safe_pkg_version,
    sha256_file,
    utc_now,
)
from scripts.deep.score_fast_pair_ddp_splits import PairModel, load_image_u8, safe_torch_load
from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    PolyUCrossPairError,
    load_polyu_cross_pairs,
    resolve_pair_image_path,
    resolve_polyu_cross_root,
)

DATASET_NAME = POLYU_CROSS_DATASET
RUN_SCHEMA_VERSION = "polyu_cross_representation_alignment_v0"

DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_CONTROLS_DIR = "artifacts/reports/diagnostics/polyu_cross_modality_controls_v0"
DEFAULT_CHECKPOINT = "artifacts/checkpoints/deep_pair_reranker_fast_ddp_anatomical_v2_ddp/best.pt"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_P2_BASELINE_CSV = "artifacts/reports/diagnostics/polyu_cross_sourceafis_readability_ladder_v0/val_comparison.csv"

CONTACT = "contact_based_2d"
CONTACTLESS = "contactless_2d"
TRAIN = "train"
VAL = "val"

PREPROCESS_ID = "fast_ddp_grayscale_resize_stretch_div255"
PREPROCESS_CONTRACT = {
    "id": PREPROCESS_ID,
    "grayscale": True,
    "resize_to": [384, 384],
    "resize_mode": "bilinear_stretch_no_pad",
    "foreground_crop": False,
    "p2_robust_intensity_norm": False,
    "invert": False,
    "value_range": "float_div255_0_1",
    "normalization": "none_after_div255",
    "channels": 1,
}

CONTROL_PROTOCOLS = (
    "contactless_to_contact_based",
    "contactless_to_contactless_same_session",
    "contactless_to_contactless_cross_session",
    "contact_based_to_contact_based_same_session",
    "contact_based_to_contact_based_cross_session",
)
WITHIN_PROTOCOLS = CONTROL_PROTOCOLS[1:]


class AlignmentError(RuntimeError):
    """Raised for protocol or artifact errors in Phase 4B.1."""


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1341
    inner_dev_fraction: float = 0.15
    projection_dim: int = 256
    temperature: float = 0.07
    batch_identities: int = 32
    max_epochs: int = 6
    patience: int = 2
    projection_lr: float = 2e-4
    encoder_lr: float = 2e-5
    weight_decay: float = 1e-4
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    eval_batch_size: int = 128
    amp: bool = True


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    rank: int
    description: str
    trains_encoder: str
    has_projection: bool


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        "R0_zero_shot_embedding",
        0,
        "Frozen SD300 ConvEncoder embedding, no projection and no training.",
        "none",
        False,
    ),
    ConditionSpec(
        "R1_projection_only",
        1,
        "Frozen SD300 ConvEncoder; train a single Linear(512, 256) projection head.",
        "none",
        True,
    ),
    ConditionSpec(
        "R2_partial_encoder_adaptation",
        2,
        "Freeze encoder.net[0:3]; train encoder.net[3] final conv block, encoder.net[6] embedding linear, and projection.",
        "final_block_and_embedding_linear",
        True,
    ),
    ConditionSpec(
        "R3_full_shared_encoder_adaptation",
        3,
        "Train the full shared SD300 ConvEncoder and the projection head.",
        "full_shared_encoder",
        True,
    ),
)
CONDITION_BY_NAME = {c.condition: c for c in CONDITIONS}
ADAPTATION_CONDITIONS = tuple(c.condition for c in CONDITIONS if c.rank > 0)
RANDOM_PROJECTION_CONTROL = "control_random_projection_untrained"
SHUFFLED_CONTROL = "control_shuffled_identity_R1"


@dataclass
class ModelBundle:
    model: "AlignmentModel"
    condition: str
    checkpoint_args: dict[str, Any]
    checkpoint_payload_meta: dict[str, Any]
    input_size: int
    encoder_dim: int
    trainable_names: list[str]
    trainable_param_count: int
    total_param_count: int


@dataclass
class TrainResult:
    condition: str
    best_epoch: int
    best_auc: float
    best_state_dict: dict[str, torch.Tensor]
    curve_rows: list[dict[str, Any]]
    best_metric_rows: list[dict[str, Any]]
    best_retrieval_rows: list[dict[str, Any]]
    best_collapse_rows: list[dict[str, Any]]
    trainable_names: list[str]
    trainable_param_count: int
    total_param_count: int


class AlignmentModel(nn.Module):
    """Shared encoder + optional single projection head for cosine scoring."""

    def __init__(self, encoder: nn.Module, *, encoder_dim: int, projection_dim: Optional[int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = int(encoder_dim)
        self.projection_dim = int(projection_dim) if projection_dim else 0
        self.projection: nn.Module
        if self.projection_dim > 0:
            self.projection = nn.Linear(self.encoder_dim, self.projection_dim)
        else:
            self.projection = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out = self.encoder(x)
        projected = self.projection(encoder_out)
        return encoder_out, projected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _encoder_out, projected = self.forward_features(x)
        return F.normalize(projected, p=2, dim=1)


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def natural_identity_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**9, text)


def set_reproducible_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def resolve_device(raw: str) -> torch.device:
    raw = str(raw).strip().lower()
    if raw in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def load_train_val_manifest(manifest_dir: Path, *, polyu_root: Optional[str]) -> tuple[pd.DataFrame, Any]:
    manifest_dir = Path(manifest_dir)
    manifest_csv = manifest_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise AlignmentError(f"Missing PolyU Cross manifest: {manifest_csv}")
    required = {
        "finger_unit_id",
        "sample_uid",
        "modality",
        "session_id",
        "split",
        "path",
    }
    frame = pd.read_csv(manifest_csv, dtype=str)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlignmentError(f"manifest.csv missing columns {missing}; found {list(frame.columns)}")

    frame = frame[frame["split"].isin([TRAIN, VAL])].copy()
    frame = frame[
        [
            "finger_unit_id",
            "sample_uid",
            "modality",
            "session_id",
            "split",
            "path",
        ]
    ].sort_values(["split", "finger_unit_id", "modality", "session_id", "sample_uid"], kind="mergesort")
    if frame.empty:
        raise AlignmentError("No TRAIN/VAL rows found in manifest.csv")

    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)
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
        raise AlignmentError(
            f"{len(missing_rows)} TRAIN/VAL manifest image(s) are missing; first={first['resolved_path']!r}"
        )
    return frame.reset_index(drop=True), resolved_root


def build_identity_pools(images: pd.DataFrame, identity_ids: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    wanted = {str(x) for x in identity_ids}
    out: dict[str, dict[str, list[str]]] = {}
    for fu, group in images[images["finger_unit_id"].astype(str).isin(wanted)].groupby("finger_unit_id", sort=False):
        cl = sorted(group[group["modality"] == CONTACTLESS]["sample_uid"].astype(str).unique().tolist())
        cb = sorted(group[group["modality"] == CONTACT]["sample_uid"].astype(str).unique().tolist())
        if cl and cb:
            out[str(fu)] = {CONTACTLESS: cl, CONTACT: cb}
    missing = sorted(wanted - set(out), key=natural_identity_key)
    if missing:
        raise AlignmentError(f"{len(missing)} identity ids lack both modalities; first={missing[:5]}")
    return out


def make_inner_split(train_identity_ids: Iterable[str], *, dev_fraction: float, seed: int) -> dict[str, list[str]]:
    ids = sorted({str(x) for x in train_identity_ids}, key=natural_identity_key)
    if len(ids) < 10:
        raise AlignmentError(f"Need at least 10 TRAIN identities for an inner split; got {len(ids)}")
    rng = np.random.default_rng(int(seed))
    order = np.asarray(ids, dtype=object)
    order = order[rng.permutation(len(order))]
    dev_count = max(1, int(round(len(ids) * float(dev_fraction))))
    inner_dev = sorted([str(x) for x in order[:dev_count]], key=natural_identity_key)
    inner_train = sorted([str(x) for x in order[dev_count:]], key=natural_identity_key)
    if set(inner_dev).intersection(inner_train):
        raise AlignmentError("Inner split is not identity-disjoint")
    return {"inner_train": inner_train, "inner_dev": inner_dev}


def choose_sample_uid(
    pools: dict[str, dict[str, list[str]]],
    identity_id: str,
    modality: str,
    *,
    epoch: int,
    seed: int,
    salt: str,
) -> str:
    values = pools[str(identity_id)][modality]
    index = stable_int(seed, epoch, identity_id, modality, salt) % len(values)
    return values[index]


def epoch_batches(
    pools: dict[str, dict[str, list[str]]],
    identity_ids: Iterable[str],
    *,
    batch_identities: int,
    epoch: int,
    seed: int,
    shuffled_identity: bool = False,
) -> list[dict[str, Any]]:
    ids = sorted([str(x) for x in identity_ids], key=natural_identity_key)
    rng = np.random.default_rng(stable_int(seed, epoch, "epoch_order") % (2**32))
    ids = [str(x) for x in np.asarray(ids, dtype=object)[rng.permutation(len(ids))]]
    batches: list[dict[str, Any]] = []
    for start in range(0, len(ids), int(batch_identities)):
        chunk = ids[start : start + int(batch_identities)]
        if len(chunk) < 2:
            continue
        cb_identity_order = list(chunk)
        if shuffled_identity:
            cb_identity_order = cb_identity_order[1:] + cb_identity_order[:1]
        cl_uids = [
            choose_sample_uid(pools, fu, CONTACTLESS, epoch=epoch, seed=seed, salt="cl")
            for fu in chunk
        ]
        cb_uids = [
            choose_sample_uid(pools, fu, CONTACT, epoch=epoch, seed=seed, salt="cb_shuf" if shuffled_identity else "cb")
            for fu in cb_identity_order
        ]
        batches.append(
            {
                "identity_ids": list(chunk),
                "cl_uids": cl_uids,
                "cb_uids": cb_uids,
                "cb_identity_ids": cb_identity_order,
            }
        )
    return batches


def _records_by_identity(images: pd.DataFrame, identity_ids: Iterable[str], modality: Optional[str] = None) -> dict[str, list[dict[str, Any]]]:
    wanted = {str(x) for x in identity_ids}
    pool = images[images["finger_unit_id"].astype(str).isin(wanted)].copy()
    if modality is not None:
        pool = pool[pool["modality"] == modality]
    out: dict[str, list[dict[str, Any]]] = {}
    for fu, group in pool.groupby("finger_unit_id", sort=False):
        rows = group.sort_values(["session_id", "sample_uid"], kind="mergesort").to_dict("records")
        out[str(fu)] = rows
    return out


def _round_robin_candidates(candidates_by_fu: dict[str, list[tuple[dict, dict]]], max_pos: int) -> list[tuple[dict, dict]]:
    order = sorted([fu for fu, values in candidates_by_fu.items() if values], key=natural_identity_key)
    cursor = {fu: 0 for fu in order}
    positives: list[tuple[dict, dict]] = []
    exhausted = False
    while len(positives) < int(max_pos) and not exhausted:
        exhausted = True
        for fu in order:
            if len(positives) >= int(max_pos):
                break
            values = candidates_by_fu[fu]
            idx = cursor[fu]
            if idx < len(values):
                positives.append(values[idx])
                cursor[fu] = idx + 1
                exhausted = False
    return positives


def _pair_row(protocol: str, split: str, pair_index: int, a: dict, b: dict, label: int) -> dict[str, Any]:
    return {
        "protocol_id": protocol,
        "split": split,
        "pair_id": f"{protocol}|{split}|{pair_index:06d}",
        "label": int(label),
        "subject_a": str(a["finger_unit_id"]),
        "subject_b": str(b["finger_unit_id"]),
        "finger_unit_a": str(a["finger_unit_id"]),
        "finger_unit_b": str(b["finger_unit_id"]),
        "sample_uid_a": str(a["sample_uid"]),
        "sample_uid_b": str(b["sample_uid"]),
        "modality_a": str(a["modality"]),
        "modality_b": str(b["modality"]),
        "session_a": str(a["session_id"]),
        "session_b": str(b["session_id"]),
        "path_a": str(a["path"]),
        "path_b": str(b["path"]),
    }


def build_cross_modal_pairs(
    images: pd.DataFrame,
    identity_ids: Iterable[str],
    *,
    split_name: str,
    max_pos: int,
    neg_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    by_fu_cl = _records_by_identity(images, identity_ids, CONTACTLESS)
    by_fu_cb = _records_by_identity(images, identity_ids, CONTACT)
    candidates: dict[str, list[tuple[dict, dict]]] = {}
    for fu in sorted(set(by_fu_cl).intersection(by_fu_cb), key=natural_identity_key):
        pairs: list[tuple[dict, dict]] = []
        for cl in by_fu_cl[fu]:
            for cb in by_fu_cb[fu]:
                pairs.append((cl, cb))
        candidates[fu] = pairs
    positives = _round_robin_candidates(candidates, max_pos)
    cb_pool = [row for rows in by_fu_cb.values() for row in rows]
    used: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for a, b in positives:
        used.add((str(a["sample_uid"]), str(b["sample_uid"])))
        rows.append(_pair_row("contactless_to_contact_based", split_name, len(rows), a, b, 1))

    rng = np.random.default_rng(stable_int(seed, split_name, "clcb_neg") % (2**32))
    for a, _b in positives:
        candidates_idx = [
            i for i, row in enumerate(cb_pool)
            if str(row["finger_unit_id"]) != str(a["finger_unit_id"])
        ]
        for idx in rng.permutation(candidates_idx):
            b = cb_pool[int(idx)]
            key = (str(a["sample_uid"]), str(b["sample_uid"]))
            if key in used:
                continue
            used.add(key)
            rows.append(_pair_row("contactless_to_contact_based", split_name, len(rows), a, b, 0))
            if sum(1 for r in rows if r["label"] == 0 and r["sample_uid_a"] == str(a["sample_uid"])) >= int(neg_per_pos):
                break
    return pd.DataFrame(rows)


def build_within_modality_pairs(
    images: pd.DataFrame,
    identity_ids: Iterable[str],
    *,
    protocol: str,
    modality: str,
    relation: str,
    split_name: str,
    max_pos: int,
    neg_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    by_fu = _records_by_identity(images, identity_ids, modality)
    candidates: dict[str, list[tuple[dict, dict]]] = {}
    for fu, rows in by_fu.items():
        pairs: list[tuple[dict, dict]] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                same = str(a["session_id"]) == str(b["session_id"])
                if relation == "same" and same:
                    pairs.append((a, b))
                elif relation == "cross" and not same:
                    pairs.append((a, b))
        candidates[fu] = pairs
    positives = _round_robin_candidates(candidates, max_pos)
    pool = [row for rows in by_fu.values() for row in rows]
    used: set[frozenset[str]] = set()
    rows: list[dict[str, Any]] = []
    for a, b in positives:
        used.add(frozenset({str(a["sample_uid"]), str(b["sample_uid"])}))
        rows.append(_pair_row(protocol, split_name, len(rows), a, b, 1))

    rng = np.random.default_rng(stable_int(seed, split_name, protocol, "neg") % (2**32))
    for a, _b in positives:
        picked = 0
        candidates_idx = []
        for i, row in enumerate(pool):
            if str(row["finger_unit_id"]) == str(a["finger_unit_id"]):
                continue
            same = str(row["session_id"]) == str(a["session_id"])
            if relation == "same" and not same:
                continue
            if relation == "cross" and same:
                continue
            candidates_idx.append(i)
        for idx in rng.permutation(candidates_idx):
            b = pool[int(idx)]
            key = frozenset({str(a["sample_uid"]), str(b["sample_uid"])})
            if key in used:
                continue
            used.add(key)
            rows.append(_pair_row(protocol, split_name, len(rows), a, b, 0))
            picked += 1
            if picked >= int(neg_per_pos):
                break
    return pd.DataFrame(rows)


def build_inner_pair_bundle(
    images: pd.DataFrame,
    identity_ids: Iterable[str],
    *,
    max_pos: int,
    neg_per_pos: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    split_name = "inner_dev"
    return {
        "contactless_to_contact_based": build_cross_modal_pairs(
            images,
            identity_ids,
            split_name=split_name,
            max_pos=max_pos,
            neg_per_pos=neg_per_pos,
            seed=seed,
        ),
        "contactless_to_contactless_same_session": build_within_modality_pairs(
            images,
            identity_ids,
            protocol="contactless_to_contactless_same_session",
            modality=CONTACTLESS,
            relation="same",
            split_name=split_name,
            max_pos=max_pos,
            neg_per_pos=neg_per_pos,
            seed=seed,
        ),
        "contactless_to_contactless_cross_session": build_within_modality_pairs(
            images,
            identity_ids,
            protocol="contactless_to_contactless_cross_session",
            modality=CONTACTLESS,
            relation="cross",
            split_name=split_name,
            max_pos=max_pos,
            neg_per_pos=neg_per_pos,
            seed=seed,
        ),
        "contact_based_to_contact_based_same_session": build_within_modality_pairs(
            images,
            identity_ids,
            protocol="contact_based_to_contact_based_same_session",
            modality=CONTACT,
            relation="same",
            split_name=split_name,
            max_pos=max_pos,
            neg_per_pos=neg_per_pos,
            seed=seed,
        ),
        "contact_based_to_contact_based_cross_session": build_within_modality_pairs(
            images,
            identity_ids,
            protocol="contact_based_to_contact_based_cross_session",
            modality=CONTACT,
            relation="cross",
            split_name=split_name,
            max_pos=max_pos,
            neg_per_pos=neg_per_pos,
            seed=seed,
        ),
    }


def load_official_val_pair_bundle(manifest_dir: Path, controls_dir: Path) -> dict[str, pd.DataFrame]:
    bundle = {
        "contactless_to_contact_based": load_polyu_cross_pairs(Path(manifest_dir) / "pairs_val.csv"),
    }
    for protocol in WITHIN_PROTOCOLS:
        path = Path(controls_dir) / "pairs" / f"pairs_{protocol}_val.csv"
        if not path.exists():
            raise AlignmentError(f"Missing official VAL control pair CSV: {path}")
        bundle[protocol] = load_polyu_cross_pairs(path)
    return bundle


def validate_pair_bundle(bundle: dict[str, pd.DataFrame], *, stage: str) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for protocol, df in bundle.items():
        required = {"pair_id", "label", "sample_uid_a", "sample_uid_b", "finger_unit_a", "finger_unit_b"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise AlignmentError(f"{stage}/{protocol} missing pair columns {missing}")
        if df["pair_id"].astype(str).duplicated().any():
            raise AlignmentError(f"{stage}/{protocol} has duplicate pair_id")
        labels = df["label"].astype(int).to_numpy()
        counts[protocol] = {
            "n_pairs": int(len(df)),
            "n_positive": int((labels == 1).sum()),
            "n_negative": int((labels == 0).sum()),
        }
        if len(np.unique(labels)) < 2:
            raise AlignmentError(f"{stage}/{protocol} lacks both classes")
    return counts


def load_image_store(images: pd.DataFrame, *, input_size: int) -> dict[str, torch.Tensor]:
    unique = images.drop_duplicates("sample_uid").sort_values("sample_uid", kind="mergesort")
    store: dict[str, torch.Tensor] = {}
    start = time.perf_counter()
    for i, row in enumerate(unique.to_dict("records")):
        if i and i % 500 == 0:
            print(f"[images] loaded {i}/{len(unique)} elapsed={time.perf_counter() - start:.1f}s", flush=True)
        store[str(row["sample_uid"])] = load_image_u8(Path(row["resolved_path"]), int(input_size))
    print(f"[images] loaded {len(store)} TRAIN/VAL images into uint8 memory cache", flush=True)
    return store


def stack_batch(image_store: dict[str, torch.Tensor], uids: list[str], *, device: torch.device) -> torch.Tensor:
    batch = torch.stack([image_store[str(uid)] for uid in uids], dim=0)
    # Keep the scientific image contract identical to the SD300 scorer
    # (uint8 grayscale -> float/255). Use standard contiguous layout here to
    # avoid large channels_last CuDNN workspaces on 8GB GPUs during R3 training.
    return batch.to(device, non_blocking=True).float().div_(255.0).contiguous()


def load_checkpoint_payload(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = safe_torch_load(Path(checkpoint))
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise AlignmentError(f"Checkpoint does not contain model_state_dict: {checkpoint}")
    args = dict(payload.get("args", {}) or {})
    meta = {
        "model_type": payload.get("model_type"),
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics"),
    }
    return payload, meta


def configure_trainability(model: AlignmentModel, condition: str) -> list[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    if condition == "R0_zero_shot_embedding":
        pass
    elif condition == "R1_projection_only":
        for param in model.projection.parameters():
            param.requires_grad_(True)
    elif condition == "R2_partial_encoder_adaptation":
        # Existing ConvEncoder: net[0..3] are conv blocks, net[6] is the
        # encoder's final embedding linear. Keep early blocks frozen.
        for param in model.encoder.net[3].parameters():  # type: ignore[attr-defined]
            param.requires_grad_(True)
        for param in model.encoder.net[6].parameters():  # type: ignore[attr-defined]
            param.requires_grad_(True)
        for param in model.projection.parameters():
            param.requires_grad_(True)
    elif condition == "R3_full_shared_encoder_adaptation":
        for param in model.encoder.parameters():
            param.requires_grad_(True)
        for param in model.projection.parameters():
            param.requires_grad_(True)
    elif condition == RANDOM_PROJECTION_CONTROL:
        pass
    else:
        raise AlignmentError(f"Unknown condition: {condition}")
    return [name for name, param in model.named_parameters() if param.requires_grad]


def set_training_modes(model: AlignmentModel, condition: str) -> None:
    model.train()
    if condition == "R1_projection_only":
        model.encoder.eval()
    elif condition == "R2_partial_encoder_adaptation":
        model.encoder.train()
        # Frozen early conv blocks stay in eval mode so BatchNorm stats do not
        # silently adapt outside the documented trainable stage.
        for idx in (0, 1, 2):
            model.encoder.net[idx].eval()  # type: ignore[attr-defined]


def build_model_bundle(
    *,
    checkpoint: Path,
    condition: str,
    projection_dim: int,
    device: torch.device,
    seed: int,
) -> ModelBundle:
    payload, meta = load_checkpoint_payload(checkpoint)
    args = dict(payload.get("args", {}) or {})
    width = int(args.get("width", 32))
    encoder_dim = int(args.get("embedding_dim", 512))
    hidden_dim = int(args.get("hidden_dim", 768))
    input_size = int(args.get("input_size", 384))

    pair_model = PairModel(width=width, embedding_dim=encoder_dim, hidden_dim=hidden_dim)
    pair_model.load_state_dict(payload["model_state_dict"], strict=True)
    encoder = pair_model.encoder

    torch.manual_seed(int(seed) + stable_int(condition, "projection") % 100_000)
    has_projection = condition not in ("R0_zero_shot_embedding",)
    if condition == RANDOM_PROJECTION_CONTROL:
        has_projection = True
    model = AlignmentModel(
        encoder,
        encoder_dim=encoder_dim,
        projection_dim=int(projection_dim) if has_projection else None,
    ).to(device)
    trainable_names = configure_trainability(model, condition)
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    return ModelBundle(
        model=model,
        condition=condition,
        checkpoint_args=args,
        checkpoint_payload_meta=meta,
        input_size=input_size,
        encoder_dim=encoder_dim,
        trainable_names=trainable_names,
        trainable_param_count=trainable,
        total_param_count=total,
    )


def symmetric_infonce(z_cl: torch.Tensor, z_cb: torch.Tensor, *, temperature: float) -> torch.Tensor:
    logits = z_cl @ z_cb.T / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


@torch.no_grad()
def extract_embeddings(
    model: AlignmentModel,
    image_store: dict[str, torch.Tensor],
    uids: list[str],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    embeddings: dict[str, np.ndarray] = {}
    pre_norms: dict[str, float] = {}
    for start in range(0, len(uids), int(batch_size)):
        chunk = uids[start : start + int(batch_size)]
        x = stack_batch(image_store, chunk, device=device)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            _enc, projected = model.forward_features(x)
            z = F.normalize(projected, p=2, dim=1)
            norms = torch.linalg.vector_norm(projected.float(), dim=1)
        z_np = z.detach().float().cpu().numpy()
        norms_np = norms.detach().float().cpu().numpy()
        for uid, emb, norm in zip(chunk, z_np, norms_np):
            embeddings[str(uid)] = emb.astype(np.float32, copy=True)
            pre_norms[str(uid)] = float(norm)
    return embeddings, pre_norms


def unique_uids_for_bundle(bundle: dict[str, pd.DataFrame], images: Optional[pd.DataFrame] = None) -> list[str]:
    uids: set[str] = set()
    for df in bundle.values():
        uids.update(df["sample_uid_a"].astype(str).tolist())
        uids.update(df["sample_uid_b"].astype(str).tolist())
    if images is not None:
        uids.update(images["sample_uid"].astype(str).tolist())
    return sorted(uids)


def auc_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return auc, eer


def group_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
    }


def score_pair_frame(df: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    scores = np.empty(len(df), dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        a = embeddings[str(getattr(row, "sample_uid_a"))]
        b = embeddings[str(getattr(row, "sample_uid_b"))]
        scores[i] = float(np.dot(a, b))
    return scores


def metric_row(
    *,
    condition: str,
    stage: str,
    protocol: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    epoch: Optional[int],
) -> dict[str, Any]:
    labels = df["label"].astype(int).to_numpy()
    auc, eer = auc_eer(labels, scores)
    gen = scores[labels == 1]
    imp = scores[labels == 0]
    gen_stats = group_stats(gen)
    imp_stats = group_stats(imp)
    return {
        "condition": condition,
        "stage": stage,
        "protocol": protocol,
        "epoch": int(epoch) if epoch is not None else "",
        "pair_count": int(len(df)),
        "genuine_count": int((labels == 1).sum()),
        "impostor_count": int((labels == 0).sum()),
        "scored_count": int(np.isfinite(scores).sum()),
        "failed_count": int((~np.isfinite(scores)).sum()),
        "roc_auc": auc,
        "eer": eer,
        "genuine_cosine_mean": gen_stats["mean"],
        "genuine_cosine_std": gen_stats["std"],
        "genuine_cosine_median": gen_stats["median"],
        "impostor_cosine_mean": imp_stats["mean"],
        "impostor_cosine_std": imp_stats["std"],
        "impostor_cosine_median": imp_stats["median"],
    }


def build_retrieval_table(images: pd.DataFrame, identity_ids: Iterable[str]) -> pd.DataFrame:
    wanted = {str(x) for x in identity_ids}
    pool = images[images["finger_unit_id"].astype(str).isin(wanted)].copy()
    rows: list[dict[str, str]] = []
    for fu, group in pool.groupby("finger_unit_id", sort=False):
        cl = group[group["modality"] == CONTACTLESS].sort_values(["session_id", "sample_uid"], kind="mergesort")
        cb = group[group["modality"] == CONTACT].sort_values(["session_id", "sample_uid"], kind="mergesort")
        if cl.empty or cb.empty:
            continue
        rows.append(
            {
                "finger_unit_id": str(fu),
                "cl_uid": str(cl.iloc[0]["sample_uid"]),
                "cb_uid": str(cb.iloc[0]["sample_uid"]),
            }
        )
    return pd.DataFrame(rows).sort_values("finger_unit_id", key=lambda s: s.map(lambda x: natural_identity_key(str(x))[0]), kind="mergesort")


def retrieval_metrics(
    *,
    condition: str,
    stage: str,
    table: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    epoch: Optional[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if table.empty:
        return rows
    ids = table["finger_unit_id"].astype(str).tolist()
    cl = np.stack([embeddings[uid] for uid in table["cl_uid"].astype(str)], axis=0)
    cb = np.stack([embeddings[uid] for uid in table["cb_uid"].astype(str)], axis=0)

    def _direction(probe: np.ndarray, gallery: np.ndarray, direction: str) -> dict[str, Any]:
        sim = probe @ gallery.T
        ranks: list[int] = []
        for i in range(sim.shape[0]):
            order = np.argsort(-sim[i], kind="mergesort")
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)
        ranks_np = np.asarray(ranks, dtype=int)
        return {
            "condition": condition,
            "stage": stage,
            "direction": direction,
            "epoch": int(epoch) if epoch is not None else "",
            "identity_count": int(len(ids)),
            "recall_at_1": float(np.mean(ranks_np <= 1)),
            "recall_at_5": float(np.mean(ranks_np <= min(5, len(ids)))),
            "mrr": float(np.mean(1.0 / ranks_np)),
        }

    rows.append(_direction(cl, cb, "CL_probe_to_CB_gallery"))
    rows.append(_direction(cb, cl, "CB_probe_to_CL_gallery"))
    return rows


def collapse_diagnostics(
    *,
    condition: str,
    stage: str,
    embeddings: dict[str, np.ndarray],
    pre_norms: dict[str, float],
    epoch: Optional[int],
) -> dict[str, Any]:
    uids = sorted(embeddings)
    mat = np.stack([embeddings[uid] for uid in uids], axis=0).astype(np.float64)
    norms = np.asarray([pre_norms[uid] for uid in uids], dtype=float)
    dim_std = np.std(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1], dtype=float)
    if mat.shape[0] > 1:
        sim = mat @ mat.T
        tri = sim[np.triu_indices(sim.shape[0], k=1)]
        mean_pairwise = float(np.mean(tri))
        near = int(np.sum(tri > 0.9999))
        total_pairs = int(tri.size)
    else:
        mean_pairwise = float("nan")
        near = 0
        total_pairs = 0
    return {
        "condition": condition,
        "stage": stage,
        "epoch": int(epoch) if epoch is not None else "",
        "embedding_count": int(mat.shape[0]),
        "embedding_dim": int(mat.shape[1]),
        "per_dim_std_mean": float(np.mean(dim_std)),
        "per_dim_std_min": float(np.min(dim_std)),
        "per_dim_std_max": float(np.max(dim_std)),
        "mean_pairwise_cosine": mean_pairwise,
        "pre_norm_mean": float(np.mean(norms)) if norms.size else float("nan"),
        "pre_norm_std": float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0,
        "pre_norm_min": float(np.min(norms)) if norms.size else float("nan"),
        "pre_norm_max": float(np.max(norms)) if norms.size else float("nan"),
        "near_identical_pairs": near,
        "near_identical_fraction": float(near / total_pairs) if total_pairs else 0.0,
    }


def evaluate_condition(
    *,
    model: AlignmentModel,
    condition: str,
    stage: str,
    pair_bundle: dict[str, pd.DataFrame],
    retrieval_table: pd.DataFrame,
    eval_images: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    device: torch.device,
    eval_batch_size: int,
    amp: bool,
    epoch: Optional[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uids = unique_uids_for_bundle(pair_bundle, eval_images)
    embeddings, pre_norms = extract_embeddings(
        model,
        image_store,
        uids,
        device=device,
        batch_size=eval_batch_size,
        amp=amp,
    )
    metric_rows = []
    for protocol in CONTROL_PROTOCOLS:
        df = pair_bundle[protocol]
        scores = score_pair_frame(df, embeddings)
        metric_rows.append(metric_row(condition=condition, stage=stage, protocol=protocol, df=df, scores=scores, epoch=epoch))
    ret_rows = retrieval_metrics(condition=condition, stage=stage, table=retrieval_table, embeddings=embeddings, epoch=epoch)
    collapse_rows = [collapse_diagnostics(condition=condition, stage=stage, embeddings=embeddings, pre_norms=pre_norms, epoch=epoch)]
    return metric_rows, ret_rows, collapse_rows


def train_condition(
    *,
    condition: str,
    checkpoint: Path,
    cfg: TrainConfig,
    inner_train_ids: list[str],
    inner_dev_pair_bundle: dict[str, pd.DataFrame],
    inner_dev_retrieval: pd.DataFrame,
    train_images: pd.DataFrame,
    inner_dev_images: pd.DataFrame,
    image_store: dict[str, torch.Tensor],
    device: torch.device,
    shuffled_identity: bool = False,
) -> TrainResult:
    bundle = build_model_bundle(
        checkpoint=checkpoint,
        condition=condition,
        projection_dim=cfg.projection_dim,
        device=device,
        seed=cfg.seed,
    )
    if bundle.trainable_param_count <= 0:
        raise AlignmentError(f"{condition} has no trainable parameters")

    pools = build_identity_pools(train_images, inner_train_ids)
    projection_params = [p for p in bundle.model.projection.parameters() if p.requires_grad]
    encoder_params = [p for name, p in bundle.model.named_parameters() if name.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if projection_params:
        param_groups.append({"params": projection_params, "lr": cfg.projection_lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.encoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")

    best_auc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: list[dict[str, Any]] = []
    best_retrieval: list[dict[str, Any]] = []
    best_collapse: list[dict[str, Any]] = []
    no_improve = 0
    curve_rows: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    for epoch in range(1, int(cfg.max_epochs) + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        set_training_modes(bundle.model, condition)
        losses: list[float] = []
        batches = epoch_batches(
            pools,
            inner_train_ids,
            batch_identities=cfg.batch_identities,
            epoch=epoch,
            seed=cfg.seed,
            shuffled_identity=shuffled_identity,
        )
        for batch in batches:
            cl = stack_batch(image_store, batch["cl_uids"], device=device)
            cb = stack_batch(image_store, batch["cb_uids"], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                z_cl = bundle.model(cl)
                z_cb = bundle.model(cb)
                loss = symmetric_infonce(z_cl, z_cb, temperature=cfg.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        metrics, retrieval, collapse = evaluate_condition(
            model=bundle.model,
            condition=condition if not shuffled_identity else SHUFFLED_CONTROL,
            stage="inner_dev",
            pair_bundle=inner_dev_pair_bundle,
            retrieval_table=inner_dev_retrieval,
            eval_images=inner_dev_images,
            image_store=image_store,
            device=device,
            eval_batch_size=cfg.eval_batch_size,
            amp=cfg.amp,
            epoch=epoch,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        clcb_auc = next(r["roc_auc"] for r in metrics if r["protocol"] == "contactless_to_contact_based")
        curve_rows.append(
            {
                "condition": condition if not shuffled_identity else SHUFFLED_CONTROL,
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "inner_dev_clcb_auc": float(clcb_auc),
                "seconds_elapsed": float(time.perf_counter() - train_start),
            }
        )
        print(
            json.dumps(
                {
                    "condition": condition if not shuffled_identity else SHUFFLED_CONTROL,
                    "epoch": epoch,
                    "train_loss": curve_rows[-1]["train_loss"],
                    "inner_dev_clcb_auc": clcb_auc,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        if math.isfinite(float(clcb_auc)) and float(clcb_auc) > best_auc + 1e-6:
            best_auc = float(clcb_auc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in bundle.model.state_dict().items()}
            best_metrics = copy.deepcopy(metrics)
            best_retrieval = copy.deepcopy(retrieval)
            best_collapse = copy.deepcopy(collapse)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(cfg.patience):
            break

    if best_state is None:
        raise AlignmentError(f"{condition} did not produce a finite inner-dev AUC")
    return TrainResult(
        condition=condition if not shuffled_identity else SHUFFLED_CONTROL,
        best_epoch=best_epoch,
        best_auc=best_auc,
        best_state_dict=best_state,
        curve_rows=curve_rows,
        best_metric_rows=best_metrics,
        best_retrieval_rows=best_retrieval,
        best_collapse_rows=best_collapse,
        trainable_names=bundle.trainable_names,
        trainable_param_count=bundle.trainable_param_count,
        total_param_count=bundle.total_param_count,
    )


def summarize_within_mean(metric_rows: list[dict[str, Any]]) -> float:
    values = [float(r["roc_auc"]) for r in metric_rows if r["protocol"] in WITHIN_PROTOCOLS]
    values = [x for x in values if math.isfinite(x)]
    return float(np.mean(values)) if values else float("nan")


def select_condition(
    inner_rows: list[dict[str, Any]],
    train_results: dict[str, TrainResult],
    *,
    within_degrade_tolerance: float = 0.03,
    auc_simplicity_margin: float = 0.02,
) -> dict[str, Any]:
    by_condition_protocol = {
        (row["condition"], row["protocol"]): row
        for row in inner_rows
        if row["condition"] in [c.condition for c in CONDITIONS]
    }
    r0_within = summarize_within_mean([row for row in inner_rows if row["condition"] == "R0_zero_shot_embedding"])
    candidates = []
    for spec in CONDITIONS:
        if spec.rank == 0:
            continue
        clcb = by_condition_protocol.get((spec.condition, "contactless_to_contact_based"))
        if clcb is None:
            continue
        rows = [row for row in inner_rows if row["condition"] == spec.condition]
        within = summarize_within_mean(rows)
        no_degrade = math.isfinite(within) and (not math.isfinite(r0_within) or within >= r0_within - within_degrade_tolerance)
        candidates.append(
            {
                "condition": spec.condition,
                "rank": spec.rank,
                "inner_dev_clcb_auc": float(clcb["roc_auc"]),
                "inner_dev_within_mean_auc": within,
                "no_material_within_degrade": bool(no_degrade),
                "best_epoch": train_results[spec.condition].best_epoch,
            }
        )
    if not candidates:
        raise AlignmentError("No adaptation candidate metrics available for selection")
    best_auc = max(c["inner_dev_clcb_auc"] for c in candidates)
    eligible = [
        c for c in candidates
        if c["inner_dev_clcb_auc"] >= best_auc - float(auc_simplicity_margin)
        and c["no_material_within_degrade"]
    ]
    if not eligible:
        eligible = [max(candidates, key=lambda c: c["inner_dev_clcb_auc"])]
    selected = sorted(eligible, key=lambda c: (int(c["rank"]), -float(c["inner_dev_clcb_auc"])))[0]
    return {
        "selection_rule": (
            "Select the simplest adaptation condition within 0.02 inner-dev CL->CB AUC of the best "
            "condition, provided its mean within-modality control AUC is not more than 0.03 below R0."
        ),
        "selected_condition": selected["condition"],
        "selected_epoch": int(selected["best_epoch"]),
        "best_inner_dev_auc": float(best_auc),
        "r0_inner_dev_within_mean_auc": r0_within,
        "candidates": candidates,
        "eligible_candidates": eligible,
    }


def load_p2_sourceafis_baseline(path: Path) -> float:
    if not Path(path).exists():
        return 0.577157
    df = pd.read_csv(path)
    row = df[df["variant"].astype(str) == "P2_robust_intensity_norm"]
    if row.empty or "clcb_auc" not in row.columns:
        return 0.577157
    return float(row.iloc[0]["clcb_auc"])


def classify_result(
    *,
    selected_condition: str,
    inner_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    p2_sourceafis_val_auc: float,
) -> dict[str, Any]:
    def _auc(condition: str, rows: list[dict[str, Any]], protocol: str = "contactless_to_contact_based") -> float:
        vals = [float(r["roc_auc"]) for r in rows if r["condition"] == condition and r["protocol"] == protocol]
        return vals[0] if vals else float("nan")

    selected_final_name = f"selected_final_{selected_condition}"
    final_selected = _auc(selected_final_name, final_rows)
    final_r0 = _auc("R0_zero_shot_embedding", final_rows)
    inner_r1 = _auc("R1_projection_only", inner_rows)
    inner_r2 = _auc("R2_partial_encoder_adaptation", inner_rows)
    inner_r3 = _auc("R3_full_shared_encoder_adaptation", inner_rows)

    selected_retrieval = [
        r for r in retrieval_rows
        if r["condition"] == selected_final_name and r["stage"] == "official_val"
    ]
    r0_retrieval = [
        r for r in retrieval_rows
        if r["condition"] == "R0_zero_shot_embedding" and r["stage"] == "official_val"
    ]
    selected_mrr = float(np.mean([r["mrr"] for r in selected_retrieval])) if selected_retrieval else float("nan")
    r0_mrr = float(np.mean([r["mrr"] for r in r0_retrieval])) if r0_retrieval else float("nan")

    verification_gain = (
        math.isfinite(final_selected)
        and math.isfinite(final_r0)
        and final_selected >= final_r0 + 0.03
        and final_selected >= float(p2_sourceafis_val_auc) + 0.02
    )
    retrieval_gain = math.isfinite(selected_mrr) and math.isfinite(r0_mrr) and selected_mrr >= r0_mrr + 0.03

    if not (verification_gain and retrieval_gain):
        label = "D. GLOBAL_ALIGNMENT_INSUFFICIENT"
    elif selected_condition == "R1_projection_only":
        label = "A. LOW_CAPACITY_ALIGNMENT_SUFFICIENT"
    elif selected_condition == "R2_partial_encoder_adaptation" and math.isfinite(inner_r3) and inner_r2 >= inner_r1 + 0.02 and abs(inner_r2 - inner_r3) <= 0.02:
        label = "B. PARTIAL_REPRESENTATION_ADAPTATION_REQUIRED"
    elif selected_condition == "R3_full_shared_encoder_adaptation":
        label = "C. FULL_END_TO_END_ADAPTATION_REQUIRED"
    else:
        label = "D. GLOBAL_ALIGNMENT_INSUFFICIENT"
    return {
        "classification": label,
        "criteria": {
            "final_selected_clcb_auc": final_selected,
            "final_r0_clcb_auc": final_r0,
            "p2_sourceafis_val_auc": float(p2_sourceafis_val_auc),
            "verification_gain_requires_selected_ge_r0_plus_0.03_and_p2_plus_0.02": bool(verification_gain),
            "selected_official_val_mean_mrr": selected_mrr,
            "r0_official_val_mean_mrr": r0_mrr,
            "retrieval_gain_requires_mean_mrr_plus_0.03": bool(retrieval_gain),
            "inner_r1_clcb_auc": inner_r1,
            "inner_r2_clcb_auc": inner_r2,
            "inner_r3_clcb_auc": inner_r3,
        },
    }


def save_experimental_checkpoint(
    path: Path,
    *,
    bundle: ModelBundle,
    selected_condition: str,
    selected_epoch: int,
    cfg: TrainConfig,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    final_metrics: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "artifact_role": "experimental_selected_alignment_model_not_canonical",
            "condition": selected_condition,
            "selected_epoch": int(selected_epoch),
            "state_dict": bundle.model.state_dict(),
            "initialization_checkpoint": str(checkpoint_path),
            "initialization_checkpoint_sha256": checkpoint_sha256,
            "trainable_parameter_count": bundle.trainable_param_count,
            "total_parameter_count": bundle.total_param_count,
            "loss": {
                "name": "symmetric_cross_modal_infonce",
                "temperature": cfg.temperature,
            },
            "preprocessing_id": PREPROCESS_ID,
            "seed": cfg.seed,
            "final_metrics": final_metrics,
        },
        path,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run(
    *,
    manifest_dir: Path,
    controls_dir: Path,
    checkpoint: Path,
    outdir: Path,
    polyu_root: Optional[str],
    device_arg: str,
    cfg: TrainConfig,
    p2_baseline_csv: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(cfg.seed)
    device = resolve_device(device_arg)

    checkpoint = Path(checkpoint)
    checkpoint_sha = sha256_file(checkpoint) or ""
    payload, checkpoint_meta = load_checkpoint_payload(checkpoint)
    ckpt_args = dict(payload.get("args", {}) or {})
    input_size = int(ckpt_args.get("input_size", 384))
    encoder_dim = int(ckpt_args.get("embedding_dim", 512))
    if input_size != 384:
        PREPROCESS_CONTRACT["resize_to"] = [input_size, input_size]

    images, resolved_root = load_train_val_manifest(manifest_dir, polyu_root=polyu_root)
    train_images = images[images["split"] == TRAIN].copy()
    val_images = images[images["split"] == VAL].copy()
    train_ids = sorted(train_images["finger_unit_id"].astype(str).unique().tolist(), key=natural_identity_key)
    val_ids = sorted(val_images["finger_unit_id"].astype(str).unique().tolist(), key=natural_identity_key)
    inner_split = make_inner_split(train_ids, dev_fraction=cfg.inner_dev_fraction, seed=cfg.seed)
    inner_train_ids = inner_split["inner_train"]
    inner_dev_ids = inner_split["inner_dev"]
    if smoke:
        inner_train_ids = inner_train_ids[: max(12, min(48, len(inner_train_ids)))]
        inner_dev_ids = inner_dev_ids[: max(8, min(16, len(inner_dev_ids)))]
        cfg = TrainConfig(
            seed=cfg.seed,
            inner_dev_fraction=cfg.inner_dev_fraction,
            projection_dim=cfg.projection_dim,
            temperature=cfg.temperature,
            batch_identities=min(cfg.batch_identities, 8),
            max_epochs=min(cfg.max_epochs, 1),
            patience=1,
            projection_lr=cfg.projection_lr,
            encoder_lr=cfg.encoder_lr,
            weight_decay=cfg.weight_decay,
            eval_max_pos=min(cfg.eval_max_pos, 40),
            eval_neg_per_pos=min(cfg.eval_neg_per_pos, 1),
            eval_batch_size=min(cfg.eval_batch_size, 32),
            amp=cfg.amp,
        )

    inner_train_images = train_images[train_images["finger_unit_id"].astype(str).isin(set(inner_train_ids))].copy()
    inner_dev_images = train_images[train_images["finger_unit_id"].astype(str).isin(set(inner_dev_ids))].copy()
    inner_bundle = build_inner_pair_bundle(
        inner_dev_images,
        inner_dev_ids,
        max_pos=cfg.eval_max_pos,
        neg_per_pos=cfg.eval_neg_per_pos,
        seed=cfg.seed,
    )
    official_val_bundle = load_official_val_pair_bundle(manifest_dir, controls_dir)
    inner_counts = validate_pair_bundle(inner_bundle, stage="inner_dev")
    official_val_counts = validate_pair_bundle(official_val_bundle, stage="official_val")
    inner_retrieval = build_retrieval_table(inner_dev_images, inner_dev_ids)
    val_retrieval = build_retrieval_table(val_images, val_ids)

    write_json(
        outdir / "experiment_config.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "training": cfg.__dict__,
            "conditions": [c.__dict__ for c in CONDITIONS],
            "random_projection_control": RANDOM_PROJECTION_CONTROL,
            "shuffled_identity_control": SHUFFLED_CONTROL,
            "preprocess_contract": PREPROCESS_CONTRACT,
            "augmentation_policy": "none",
            "optimizer": {
                "family": "AdamW",
                "projection_lr": cfg.projection_lr,
                "encoder_lr": cfg.encoder_lr,
                "weight_decay": cfg.weight_decay,
                "broad_hyperparameter_search": False,
            },
            "selection_rule": (
                "Simplest R1/R2/R3 within 0.02 inner-dev CL->CB AUC of the best, "
                "with mean within-modality control AUC no more than 0.03 below R0."
            ),
        },
    )
    write_json(
        outdir / "inner_split.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "seed": cfg.seed,
            "source": "official TRAIN finger_unit_ids only",
            "inner_train_count": len(inner_train_ids),
            "inner_dev_count": len(inner_dev_ids),
            "inner_train": inner_train_ids,
            "inner_dev": inner_dev_ids,
            "identity_disjoint": not bool(set(inner_train_ids).intersection(inner_dev_ids)),
            "official_val_identity_count": len(val_ids),
            "test_used": False,
        },
    )

    image_store = load_image_store(images, input_size=input_size)

    condition_inventory: list[dict[str, Any]] = []
    for spec in CONDITIONS:
        bundle = build_model_bundle(
            checkpoint=checkpoint,
            condition=spec.condition,
            projection_dim=cfg.projection_dim,
            device=device,
            seed=cfg.seed,
        )
        condition_inventory.append(
            {
                "condition": spec.condition,
                "rank": spec.rank,
                "description": spec.description,
                "trainable_policy": spec.trains_encoder,
                "projection_dim": cfg.projection_dim if spec.has_projection else 0,
                "encoder_dim": encoder_dim,
                "trainable_param_count": bundle.trainable_param_count,
                "total_param_count": bundle.total_param_count,
                "trainable_parameter_names": ";".join(bundle.trainable_names),
            }
        )
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    random_bundle = build_model_bundle(
        checkpoint=checkpoint,
        condition=RANDOM_PROJECTION_CONTROL,
        projection_dim=cfg.projection_dim,
        device=device,
        seed=cfg.seed,
    )
    condition_inventory.append(
        {
            "condition": RANDOM_PROJECTION_CONTROL,
            "rank": -1,
            "description": "Frozen SD300 encoder with random untrained Linear(512, 256) projection; evaluated before training.",
            "trainable_policy": "none_evaluated_only",
            "projection_dim": cfg.projection_dim,
            "encoder_dim": encoder_dim,
            "trainable_param_count": random_bundle.trainable_param_count,
            "total_param_count": random_bundle.total_param_count,
            "trainable_parameter_names": "",
        }
    )
    write_csv(outdir / "condition_inventory.csv", condition_inventory)

    inner_metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    training_curve_rows: list[dict[str, Any]] = []

    # R0 required baseline: frozen encoder embeddings before the pair head.
    r0_bundle = build_model_bundle(
        checkpoint=checkpoint,
        condition="R0_zero_shot_embedding",
        projection_dim=cfg.projection_dim,
        device=device,
        seed=cfg.seed,
    )
    rows, ret, collapse = evaluate_condition(
        model=r0_bundle.model,
        condition="R0_zero_shot_embedding",
        stage="inner_dev",
        pair_bundle=inner_bundle,
        retrieval_table=inner_retrieval,
        eval_images=inner_dev_images,
        image_store=image_store,
        device=device,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
        epoch=None,
    )
    inner_metric_rows.extend(rows)
    retrieval_rows.extend(ret)
    collapse_rows.extend(collapse)
    print("[R0] inner-dev evaluation complete", flush=True)

    rows, ret, collapse = evaluate_condition(
        model=random_bundle.model,
        condition=RANDOM_PROJECTION_CONTROL,
        stage="inner_dev",
        pair_bundle=inner_bundle,
        retrieval_table=inner_retrieval,
        eval_images=inner_dev_images,
        image_store=image_store,
        device=device,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
        epoch=None,
    )
    inner_metric_rows.extend(rows)
    retrieval_rows.extend(ret)
    collapse_rows.extend(collapse)
    del random_bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_results: dict[str, TrainResult] = {}
    for condition in ADAPTATION_CONDITIONS:
        print(f"[train] {condition}", flush=True)
        result = train_condition(
            condition=condition,
            checkpoint=checkpoint,
            cfg=cfg,
            inner_train_ids=inner_train_ids,
            inner_dev_pair_bundle=inner_bundle,
            inner_dev_retrieval=inner_retrieval,
            train_images=inner_train_images,
            inner_dev_images=inner_dev_images,
            image_store=image_store,
            device=device,
        )
        train_results[condition] = result
        inner_metric_rows.extend(result.best_metric_rows)
        retrieval_rows.extend(result.best_retrieval_rows)
        collapse_rows.extend(result.best_collapse_rows)
        training_curve_rows.extend(result.curve_rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Required shuffled-identity diagnostic, intentionally small and R1-only.
    shuffled_cfg = TrainConfig(
        seed=cfg.seed + 17,
        inner_dev_fraction=cfg.inner_dev_fraction,
        projection_dim=cfg.projection_dim,
        temperature=cfg.temperature,
        batch_identities=cfg.batch_identities,
        max_epochs=min(2, cfg.max_epochs),
        patience=1,
        projection_lr=cfg.projection_lr,
        encoder_lr=cfg.encoder_lr,
        weight_decay=cfg.weight_decay,
        eval_max_pos=cfg.eval_max_pos,
        eval_neg_per_pos=cfg.eval_neg_per_pos,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
    )
    shuffled = train_condition(
        condition="R1_projection_only",
        checkpoint=checkpoint,
        cfg=shuffled_cfg,
        inner_train_ids=inner_train_ids,
        inner_dev_pair_bundle=inner_bundle,
        inner_dev_retrieval=inner_retrieval,
        train_images=inner_train_images,
        inner_dev_images=inner_dev_images,
        image_store=image_store,
        device=device,
        shuffled_identity=True,
    )
    inner_metric_rows.extend(shuffled.best_metric_rows)
    retrieval_rows.extend(shuffled.best_retrieval_rows)
    collapse_rows.extend(shuffled.best_collapse_rows)
    training_curve_rows.extend(shuffled.curve_rows)

    write_csv(outdir / "inner_dev_metrics.csv", inner_metric_rows)
    write_csv(outdir / "training_curves.csv", training_curve_rows)

    selection = select_condition(inner_metric_rows, train_results)
    write_json(outdir / "selection_decision.json", selection)

    # Final fit: reinitialize from canonical checkpoint and train selected
    # condition once on all official TRAIN identities for the selected epoch.
    selected_condition = selection["selected_condition"]
    selected_epoch = int(selection["selected_epoch"])
    final_cfg = TrainConfig(
        seed=cfg.seed + 101,
        inner_dev_fraction=cfg.inner_dev_fraction,
        projection_dim=cfg.projection_dim,
        temperature=cfg.temperature,
        batch_identities=cfg.batch_identities,
        max_epochs=selected_epoch,
        patience=selected_epoch + 1,
        projection_lr=cfg.projection_lr,
        encoder_lr=cfg.encoder_lr,
        weight_decay=cfg.weight_decay,
        eval_max_pos=cfg.eval_max_pos,
        eval_neg_per_pos=cfg.eval_neg_per_pos,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
    )
    final_bundle = build_model_bundle(
        checkpoint=checkpoint,
        condition=selected_condition,
        projection_dim=cfg.projection_dim,
        device=device,
        seed=final_cfg.seed,
    )
    final_pools = build_identity_pools(train_images, train_ids)
    projection_params = [p for p in final_bundle.model.projection.parameters() if p.requires_grad]
    encoder_params = [p for name, p in final_bundle.model.named_parameters() if name.startswith("encoder.") and p.requires_grad]
    groups = []
    if projection_params:
        groups.append({"params": projection_params, "lr": cfg.projection_lr})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": cfg.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")
    for epoch in range(1, selected_epoch + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        set_training_modes(final_bundle.model, selected_condition)
        losses: list[float] = []
        for batch in epoch_batches(
            final_pools,
            train_ids,
            batch_identities=cfg.batch_identities,
            epoch=epoch,
            seed=final_cfg.seed,
        ):
            cl = stack_batch(image_store, batch["cl_uids"], device=device)
            cb = stack_batch(image_store, batch["cb_uids"], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                loss = symmetric_infonce(final_bundle.model(cl), final_bundle.model(cb), temperature=cfg.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        training_curve_rows.append(
            {
                "condition": f"selected_final_{selected_condition}",
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "inner_dev_clcb_auc": "",
                "seconds_elapsed": "",
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"condition": f"selected_final_{selected_condition}", "epoch": epoch, "train_loss": float(np.mean(losses))}, ensure_ascii=True), flush=True)

    # Official VAL is evaluated here, after selection is written.
    final_val_rows: list[dict[str, Any]] = []
    r0_val_rows, r0_val_ret, r0_val_collapse = evaluate_condition(
        model=r0_bundle.model,
        condition="R0_zero_shot_embedding",
        stage="official_val",
        pair_bundle=official_val_bundle,
        retrieval_table=val_retrieval,
        eval_images=val_images,
        image_store=image_store,
        device=device,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
        epoch=None,
    )
    final_val_rows.extend(r0_val_rows)
    retrieval_rows.extend(r0_val_ret)
    collapse_rows.extend(r0_val_collapse)

    selected_val_rows, selected_ret, selected_collapse = evaluate_condition(
        model=final_bundle.model,
        condition=f"selected_final_{selected_condition}",
        stage="official_val",
        pair_bundle=official_val_bundle,
        retrieval_table=val_retrieval,
        eval_images=val_images,
        image_store=image_store,
        device=device,
        eval_batch_size=cfg.eval_batch_size,
        amp=cfg.amp,
        epoch=selected_epoch,
    )
    final_val_rows.extend(selected_val_rows)
    retrieval_rows.extend(selected_ret)
    collapse_rows.extend(selected_collapse)

    p2_baseline = load_p2_sourceafis_baseline(p2_baseline_csv)
    classification = classify_result(
        selected_condition=selected_condition,
        inner_rows=inner_metric_rows,
        final_rows=final_val_rows,
        retrieval_rows=retrieval_rows,
        p2_sourceafis_val_auc=p2_baseline,
    )

    write_csv(outdir / "final_val_metrics.csv", final_val_rows)
    write_csv(outdir / "retrieval_metrics.csv", retrieval_rows)
    write_csv(outdir / "embedding_collapse_diagnostics.csv", collapse_rows)
    write_csv(outdir / "training_curves.csv", training_curve_rows)

    experimental_ckpt = outdir / "experimental_checkpoints" / f"{selected_condition}_final.pt"
    save_experimental_checkpoint(
        experimental_ckpt,
        bundle=final_bundle,
        selected_condition=selected_condition,
        selected_epoch=selected_epoch,
        cfg=cfg,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        final_metrics=selected_val_rows,
    )

    canonical_files = {
        "manifest_csv": Path(manifest_dir) / "manifest.csv",
        "pairs_train_csv": Path(manifest_dir) / "pairs_train.csv",
        "pairs_val_csv": Path(manifest_dir) / "pairs_val.csv",
        "pairs_test_csv_integrity_only": Path(manifest_dir) / "pairs_test.csv",
        "checkpoint": checkpoint,
    }
    artifact_hashes = {name: sha256_file(path) for name, path in canonical_files.items() if path.exists()}
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_args": ckpt_args,
        "encoder": {
            "class": "scripts.deep.score_fast_pair_ddp_splits.ConvEncoder",
            "embedding_dim": encoder_dim,
            "shared_for_cl_and_cb": True,
            "pair_head_used_for_scoring": False,
        },
        "preprocess_contract": PREPROCESS_CONTRACT,
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "identity_counts": {
            "official_train": len(train_ids),
            "inner_train": len(inner_train_ids),
            "inner_dev": len(inner_dev_ids),
            "official_val": len(val_ids),
        },
        "pair_counts": {"inner_dev": inner_counts, "official_val": official_val_counts},
        "outputs": {
            "experiment_config_json": str(outdir / "experiment_config.json"),
            "inner_split_json": str(outdir / "inner_split.json"),
            "condition_inventory_csv": str(outdir / "condition_inventory.csv"),
            "inner_dev_metrics_csv": str(outdir / "inner_dev_metrics.csv"),
            "training_curves_csv": str(outdir / "training_curves.csv"),
            "selection_decision_json": str(outdir / "selection_decision.json"),
            "final_val_metrics_csv": str(outdir / "final_val_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "embedding_collapse_diagnostics_csv": str(outdir / "embedding_collapse_diagnostics.csv"),
            "experimental_checkpoint": str(experimental_ckpt),
        },
        "classification": classification,
        "canonical_artifact_sha256": artifact_hashes,
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
            "trained_on_official_train_only": True,
            "official_val_used_for_early_stopping_or_selection": False,
            "official_val_evaluated_once_after_selection": True,
            "test_pairs_scored": False,
            "test_images_loaded": False,
            "modified_manifest_or_pairs": False,
            "modified_checkpoint": False,
            "used_p2_preprocessing_primary": False,
            "used_fusion_scores_as_targets": False,
            "used_identity_classifier_softmax_primary_objective": False,
            "used_context_features_as_model_inputs": False,
            "separate_modality_encoders": False,
            "broad_hyperparameter_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)

    return {
        "outdir": outdir,
        "selection": selection,
        "classification": classification,
        "final_val_metrics": pd.DataFrame(final_val_rows),
        "inner_dev_metrics": pd.DataFrame(inner_metric_rows),
        "retrieval_metrics": pd.DataFrame(retrieval_rows),
        "checkpoint_sha256": checkpoint_sha,
        "condition_inventory": pd.DataFrame(condition_inventory),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4B.1 PolyU Cross representation alignment diagnostic.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--controls_dir", type=str, default=DEFAULT_CONTROLS_DIR)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--polyu_root", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--projection_dim", type=int, default=TrainConfig.projection_dim)
    p.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    p.add_argument("--batch_identities", type=int, default=TrainConfig.batch_identities)
    p.add_argument("--max_epochs", type=int, default=TrainConfig.max_epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--projection_lr", type=float, default=TrainConfig.projection_lr)
    p.add_argument("--encoder_lr", type=float, default=TrainConfig.encoder_lr)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--eval_max_pos", type=int, default=TrainConfig.eval_max_pos)
    p.add_argument("--eval_neg_per_pos", type=int, default=TrainConfig.eval_neg_per_pos)
    p.add_argument("--eval_batch_size", type=int, default=TrainConfig.eval_batch_size)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--p2_baseline_csv", type=str, default=DEFAULT_P2_BASELINE_CSV)
    p.add_argument("--smoke", action="store_true", help="Fast protocol smoke: one epoch and smaller diagnostic pairs.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        seed=int(args.seed),
        projection_dim=int(args.projection_dim),
        temperature=float(args.temperature),
        batch_identities=int(args.batch_identities),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        projection_lr=float(args.projection_lr),
        encoder_lr=float(args.encoder_lr),
        weight_decay=float(args.weight_decay),
        eval_max_pos=int(args.eval_max_pos),
        eval_neg_per_pos=int(args.eval_neg_per_pos),
        eval_batch_size=int(args.eval_batch_size),
        amp=not bool(args.no_amp),
    )
    try:
        result = run(
            manifest_dir=resolve_repo_path(args.data_dir),
            controls_dir=resolve_repo_path(args.controls_dir),
            checkpoint=resolve_repo_path(args.checkpoint),
            outdir=resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            device_arg=str(args.device),
            cfg=cfg,
            p2_baseline_csv=resolve_repo_path(args.p2_baseline_csv),
            smoke=bool(args.smoke),
        )
    except (AlignmentError, PolyUCrossPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n=== PolyU Cross representation-alignment Phase 4B.1 complete ===")
    print(f"Output dir          : {result['outdir']}")
    print(f"Checkpoint SHA256   : {result['checkpoint_sha256']}")
    print(f"Selected condition  : {result['selection']['selected_condition']} @ epoch {result['selection']['selected_epoch']}")
    print(f"Classification      : {result['classification']['classification']}")
    final = result["final_val_metrics"]
    print("\nOfficial VAL verification AUC:")
    print(final[["condition", "protocol", "roc_auc", "eer"]].to_string(index=False))
    print("\nOfficial VAL retrieval:")
    ret = result["retrieval_metrics"]
    print(ret[ret["stage"] == "official_val"][["condition", "direction", "recall_at_1", "recall_at_5", "mrr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
