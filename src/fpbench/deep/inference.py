from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import build_pair_model


def _list_from_batch_meta(meta: dict[str, Any], key: str, n: int) -> list[Any]:
    value = meta.get(key, [""] * n)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


@torch.no_grad()
def score_pair_dataloader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    method: str = "deep_pair_reranker_v1",
    amp: bool = False,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        meta = batch.get("meta", {})
        n = int(len(labels))
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            logits = model(image_a, image_b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
        logits_list = logits.detach().cpu().numpy().astype(float).tolist()
        per_pair_ms = elapsed_ms / max(n, 1)
        meta_lists = {key: _list_from_batch_meta(meta, key, n) for key in [
            "dataset", "split", "pair_id", "subject_a", "subject_b", "finger_position", "frgp", "path_a", "path_b"
        ]}
        for i in range(n):
            rows.append(
                {
                    "method": method,
                    "dataset": meta_lists["dataset"][i],
                    "split": meta_lists["split"][i],
                    "pair_id": meta_lists["pair_id"][i],
                    "label": int(labels[i]),
                    "subject_a": meta_lists["subject_a"][i],
                    "subject_b": meta_lists["subject_b"][i],
                    "finger_position": meta_lists["finger_position"][i],
                    "frgp": meta_lists["frgp"][i],
                    "path_a": meta_lists["path_a"][i],
                    "path_b": meta_lists["path_b"][i],
                    "score": float(probs[i]),
                    "score_semantics": "deep_pair_match_probability",
                    "higher_is_more_similar": True,
                    "logit": float(logits_list[i]),
                    "probability": float(probs[i]),
                    "pair_total_ms": float(per_pair_ms),
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    config = dict(payload.get("config", {}))
    model_config = dict(config.get("model", config.get("model_config", {})))
    model = build_pair_model(
        backbone=model_config.get("backbone", "small_cnn"),
        channels=int(model_config.get("channels", config.get("channels", 1))),
        embedding_dim=int(model_config.get("embedding_dim", 256)),
        hidden_dim=int(model_config.get("hidden_dim", 512)),
        dropout=float(model_config.get("dropout", 0.15)),
        pretrained=False,
    )
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state)
    return model, config
