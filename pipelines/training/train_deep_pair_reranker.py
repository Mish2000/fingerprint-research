from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.evaluate import compute_auc_eer
from src.fpbench.deep import FingerprintPairDataset
from src.fpbench.deep.models import build_pair_model
from src.fpbench.deep.samplers import build_weighted_random_sampler
from src.fpbench.deep.train_utils import (
    dataloader_worker_count,
    resolve_device,
    set_reproducible_seed,
    utc_now,
    write_json,
)
from src.fpbench.deep.transforms import FingerprintPairTransform

METHOD = "deep_pair_reranker_v1"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _pairs_path(repo_root: Path, dataset: str, split: str) -> Path:
    candidates = [
        repo_root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_dataset(
    *,
    repo_root: Path,
    data_root: Path | None,
    datasets: tuple[str, ...],
    split: str,
    transform: FingerprintPairTransform,
    limit_per_dataset: int,
    image_cache_size: int,
) -> ConcatDataset:
    parts = []
    for dataset in datasets:
        pairs_csv = _pairs_path(repo_root, dataset, split)
        parts.append(
            FingerprintPairDataset(
                pairs_csv,
                dataset=dataset,
                split=split,
                repo_root=repo_root,
                data_root=data_root,
                transform=transform,
                limit=int(limit_per_dataset),
                image_cache_size=int(image_cache_size),
            )
        )
    return ConcatDataset(parts)


def _labels_from_concat(dataset: ConcatDataset) -> list[int]:
    labels: list[int] = []
    for part in dataset.datasets:
        labels.extend(getattr(part, "labels"))
    return labels


def _loader_kwargs(*, workers: int, device: torch.device, prefetch_factor: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(workers),
        "pin_memory": device.type == "cuda",
    }
    if int(workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, amp: bool) -> dict[str, float | int]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            logits = model(image_a, image_b)
            loss_values = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
        scores_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
        losses.extend(loss_values.detach().cpu().numpy().astype(float).tolist())
    labels_np = np.asarray(labels_all, dtype=int)
    scores_np = np.asarray(scores_all, dtype=float)
    auc_eer = compute_auc_eer(labels_np, scores_np)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": float(auc_eer.auc),
        "eer": float(auc_eer.eer),
        "eer_threshold": float(auc_eer.eer_threshold),
        "n_pairs": int(labels_np.size),
        "n_positive": int(np.sum(labels_np == 1)),
        "n_negative": int(np.sum(labels_np == 0)),
    }


def _save_checkpoint(path: Path, *, model: nn.Module, config: dict[str, Any], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "deep_pair_reranker_checkpoint_v1",
            "method": METHOD,
            "created_at": utc_now(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2B deep pair reranker prototype.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None, help="Optional remapping root for raw image paths, useful on Kaggle/Linux.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints" / METHOD)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--backbone", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported. Avoid in offline Kaggle runs unless cached.")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=0,
        help="Per-worker LRU cache size for transformed fingerprint tensors. 0 disables caching.",
    )
    parser.add_argument("--no-foreground-crop", action="store_true", help="Skip CPU foreground crop during preprocessing.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit-train-pairs-per-dataset", type=int, default=0, help="Debug limit. 0 means all train pairs.")
    parser.add_argument("--limit-val-pairs-per-dataset", type=int, default=0, help="Debug limit. 0 means all val pairs.")
    parser.add_argument("--debug", action="store_true", help="Small fast smoke-training mode.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    datasets = _parse_csv_list(args.datasets)
    if args.debug:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 8)
        args.limit_train_pairs_per_dataset = args.limit_train_pairs_per_dataset or 64
        args.limit_val_pairs_per_dataset = args.limit_val_pairs_per_dataset or 32

    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    workers = dataloader_worker_count(args.num_workers)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    foreground_crop = not bool(args.no_foreground_crop)
    transform = FingerprintPairTransform(size=int(args.input_size), channels=int(args.channels), foreground_crop=foreground_crop)
    train_dataset = _build_dataset(
        repo_root=repo_root,
        data_root=data_root,
        datasets=datasets,
        split="train",
        transform=transform,
        limit_per_dataset=int(args.limit_train_pairs_per_dataset),
        image_cache_size=int(args.image_cache_size),
    )
    val_dataset = _build_dataset(
        repo_root=repo_root,
        data_root=data_root,
        datasets=datasets,
        split="val",
        transform=transform,
        limit_per_dataset=int(args.limit_val_pairs_per_dataset),
        image_cache_size=int(args.image_cache_size),
    )

    train_labels = _labels_from_concat(train_dataset)
    train_sampler = build_weighted_random_sampler(train_labels, num_samples=len(train_labels), seed=int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        **_loader_kwargs(workers=workers, device=device, prefetch_factor=int(args.prefetch_factor)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        **_loader_kwargs(workers=workers, device=device, prefetch_factor=int(args.prefetch_factor)),
    )

    model = build_pair_model(
        backbone=args.backbone,
        channels=int(args.channels),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        pretrained=bool(args.pretrained),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    config = {
        "schema_version": "deep_pair_reranker_training_config_v1",
        "method": METHOD,
        "repo_root": str(repo_root),
        "data_root": str(data_root) if data_root is not None else "",
        "datasets": list(datasets),
        "train_split": "train",
        "val_split": "val",
        "model": {
            "backbone": args.backbone,
            "channels": int(args.channels),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "pretrained": bool(args.pretrained),
        },
        "preprocess": {
            "input_size": int(args.input_size),
            "channels": int(args.channels),
            "foreground_crop": foreground_crop,
            "image_cache_size": int(args.image_cache_size),
            "normalization": "pixel_to_0_1_then_mean0.5_std0.5",
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "sampler": "inverse_frequency_weighted_random_sampler_train_only",
            "loss": "BCEWithLogitsLoss",
            "seed": int(args.seed),
            "amp": bool(args.amp),
            "device": str(device),
            "num_workers": int(workers),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": int(workers) > 0,
        },
        "protocol_guards": {
            "fit_split": "train only",
            "model_selection_split": "val only",
            "test_usage": "never during training",
        },
    }
    write_json(output_dir / "config.json", config)

    history: list[dict[str, Any]] = []
    best_auc = -math.inf
    start_run = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        start_epoch = time.perf_counter()
        for batch in train_loader:
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                logits = model(image_a, image_b)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_metrics = _evaluate(model, val_loader, device=device, amp=bool(args.amp))
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_eer": val_metrics["eer"],
            "seconds": float(time.perf_counter() - start_epoch),
        }
        history.append(row)
        _save_checkpoint(output_dir / "last.pt", model=model, config=config, epoch=epoch, metrics=row)
        if math.isfinite(float(val_metrics["auc"])) and float(val_metrics["auc"]) > best_auc:
            best_auc = float(val_metrics["auc"])
            _save_checkpoint(output_dir / "best.pt", model=model, config=config, epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False))

    manifest = {
        "schema_version": "deep_pair_reranker_training_manifest_v1",
        "created_at": utc_now(),
        "method": METHOD,
        "config_json": str(output_dir / "config.json"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "history": history,
        "total_runtime_seconds": float(time.perf_counter() - start_run),
        "train_pairs": int(len(train_dataset)),
        "val_pairs": int(len(val_dataset)),
        "train_positive": int(sum(1 for x in train_labels if int(x) == 1)),
        "train_negative": int(sum(1 for x in train_labels if int(x) == 0)),
    }
    write_json(output_dir / "training_manifest.json", manifest)
    print(f"[OK] wrote {output_dir}")


if __name__ == "__main__":
    main()
