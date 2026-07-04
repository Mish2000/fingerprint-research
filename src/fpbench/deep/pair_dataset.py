from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset

from .image_io import load_grayscale_image
from .transforms import FingerprintPairTransform


@dataclass(frozen=True)
class PairDatasetConfig:
    pairs_csv: str
    dataset: str = ""
    split: str = ""
    repo_root: str = ""
    data_root: str = ""
    input_size: int = 384
    channels: int = 1
    foreground_crop: bool = True
    limit: int = 0
    image_cache_size: int = 0


class FingerprintPairDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset over canonical plain/roll pair CSV files.

    Expected columns include at least ``pair_id``, ``label``, ``path_a`` and
    ``path_b``. All traceability fields are preserved in the returned sample.
    """

    def __init__(
        self,
        pairs_csv: str | Path,
        *,
        dataset: str | None = None,
        split: str | None = None,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
        transform: FingerprintPairTransform | None = None,
        limit: int = 0,
        require_existing_images: bool = False,
        image_cache_size: int = 0,
    ) -> None:
        self.pairs_csv = Path(pairs_csv)
        if not self.pairs_csv.exists():
            raise FileNotFoundError(f"pairs CSV not found: {self.pairs_csv}")
        self.repo_root = Path(repo_root).resolve() if repo_root is not None and str(repo_root) else None
        self.data_root = Path(data_root).resolve() if data_root is not None and str(data_root) else None
        self.transform = transform or FingerprintPairTransform()
        self.require_existing_images = bool(require_existing_images)
        self.image_cache_size = max(0, int(image_cache_size))
        self._image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        df = pd.read_csv(self.pairs_csv, dtype={"pair_id": str, "subject_a": str, "subject_b": str, "frgp": str, "finger_id": str, "finger_position": str, "path_a": str, "path_b": str})
        required = {"pair_id", "label", "path_a", "path_b"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{self.pairs_csv} missing required columns: {missing}")

        df = df.copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError(f"{self.pairs_csv} label column must contain only 0/1")

        if split:
            if "split" not in df.columns:
                raise ValueError(f"split={split!r} requested but {self.pairs_csv} has no split column")
            df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()
        if dataset:
            df["dataset"] = str(dataset)
        elif "dataset" not in df.columns:
            df["dataset"] = ""
        if split and "split" not in df.columns:
            df["split"] = str(split)
        elif "split" not in df.columns:
            df["split"] = ""

        if "finger_position" not in df.columns:
            if "frgp" in df.columns:
                df["finger_position"] = df["frgp"].astype(str)
            elif "finger_id" in df.columns:
                df["finger_position"] = df["finger_id"].astype(str)
            else:
                df["finger_position"] = ""
        if "frgp" not in df.columns:
            df["frgp"] = df["finger_position"].astype(str)
        for col in ("subject_a", "subject_b"):
            if col not in df.columns:
                df[col] = ""
        if int(limit) > 0:
            df = df.head(int(limit)).copy()
        if df.empty:
            raise ValueError(f"No rows available in {self.pairs_csv} after split/limit filtering")
        self.df = df.reset_index(drop=True)

    @property
    def labels(self) -> list[int]:
        return [int(v) for v in self.df["label"].tolist()]

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_transformed(self, raw_path: str) -> torch.Tensor:
        cache_key = str(raw_path)
        if self.image_cache_size > 0:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                return cached

        image = load_grayscale_image(raw_path, repo_root=self.repo_root, data_root=self.data_root)
        tensor = self.transform(image)
        if self.image_cache_size > 0:
            self._image_cache[cache_key] = tensor
            self._image_cache.move_to_end(cache_key)
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return tensor

    def metadata_for_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        keys = [
            "dataset",
            "split",
            "pair_id",
            "label",
            "subject_a",
            "subject_b",
            "finger_position",
            "frgp",
            "path_a",
            "path_b",
        ]
        return {key: row.get(key, "") for key in keys}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[int(index)].to_dict()
        tensor_a = self._load_transformed(row["path_a"])
        tensor_b = self._load_transformed(row["path_b"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = self.metadata_for_row(row)
        return {
            "image_a": tensor_a,
            "image_b": tensor_b,
            "label": label,
            "meta": meta,
        }
