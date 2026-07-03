from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("torch")

from src.fpbench.deep.image_io import resolve_fingerprint_path
from src.fpbench.deep.pair_dataset import FingerprintPairDataset
from src.fpbench.deep.transforms import FingerprintPairTransform


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 48), color=200).save(path)


def test_windows_path_resolver_remaps_data_suffix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    image_path = repo / "data" / "raw" / "NIST" / "sd300b" / "images" / "plain" / "a.png"
    _write_image(image_path)
    raw = r"C:\fingerprint-research\data\raw\NIST\sd300b\images\plain\a.png"
    resolved = resolve_fingerprint_path(raw, repo_root=repo)
    assert resolved == image_path.resolve()


def test_pair_dataset_loads_pair_and_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    a = repo / "data" / "raw" / "NIST" / "sd300b" / "images" / "plain" / "a.png"
    b = repo / "data" / "raw" / "NIST" / "sd300b" / "images" / "roll" / "b.png"
    _write_image(a)
    _write_image(b)
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame([
        {
            "dataset": "nist_sd300b",
            "split": "train",
            "pair_id": "p1",
            "label": 1,
            "subject_a": "0001",
            "subject_b": "0001",
            "frgp": "02",
            "path_a": str(a),
            "path_b": str(b),
        }
    ]).to_csv(pairs, index=False)
    ds = FingerprintPairDataset(
        pairs,
        dataset="nist_sd300b",
        split="train",
        repo_root=repo,
        transform=FingerprintPairTransform(size=32, channels=1),
    )
    item = ds[0]
    assert tuple(item["image_a"].shape) == (1, 32, 32)
    assert item["label"].item() == 1.0
    assert item["meta"]["pair_id"] == "p1"
    assert item["meta"]["finger_position"] == "02"
