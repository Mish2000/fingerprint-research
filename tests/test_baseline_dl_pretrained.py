from __future__ import annotations

import builtins
import inspect
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.fpbench.matchers import baseline_dl
from src.fpbench.matchers.baseline_dl import (
    BaselineDL,
    DLBaselineConfig,
    PretrainedEmbedder,
    PretrainedModelUnavailableError,
)


class _Weights:
    DEFAULT = object()


class _DummyResNet(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Identity(),
        )

    def children(self):
        return iter(self.features)


class _DummyVit(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 768), dtype=x.dtype, device=x.device)


def _install_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    tv = types.ModuleType("torchvision")
    tvm = types.ModuleType("torchvision.models")
    tvm.ResNet18_Weights = _Weights
    tvm.ResNet50_Weights = _Weights
    tvm.ViT_B_16_Weights = _Weights
    tvm.resnet18 = lambda weights=None: _DummyResNet(512)
    tvm.resnet50 = lambda weights=None: _DummyResNet(2048)
    tvm.vit_b_16 = lambda weights=None: _DummyVit()
    tv.models = tvm
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.models", tvm)


def test_pretrained_embedder_raises_when_torchvision_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fail_torchvision_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name).startswith("torchvision"):
            raise RuntimeError("pretrained weights unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_torchvision_import)

    with pytest.raises(
        PretrainedModelUnavailableError,
        match="backbone='resnet18'.*RuntimeError: pretrained weights unavailable",
    ):
        PretrainedEmbedder("resnet18")


def test_pretrained_embedder_has_no_random_128_dim_fallback_path() -> None:
    source = inspect.getsource(baseline_dl.PretrainedEmbedder)

    assert "dim = 128" not in source
    assert "Conv2d(3, 16" not in source
    assert "Linear(64, dim" not in source


@pytest.mark.parametrize(
    ("backbone", "expected_dim"),
    [
        ("resnet18", 512),
        ("resnet50", 2048),
        ("vit_base", 768),
    ],
)
def test_pretrained_embedder_reports_contract_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    backbone: str,
    expected_dim: int,
) -> None:
    _install_fake_torchvision(monkeypatch)

    model = PretrainedEmbedder(backbone)
    output = model(torch.zeros((2, 3, 224, 224), dtype=torch.float32))

    assert model.embed_dim == expected_dim
    assert output.shape == (2, expected_dim)


def test_baseline_dl_rejects_model_embed_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class WrongDimEmbedder(nn.Module):
        embed_dim = 128

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 128), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(baseline_dl, "PretrainedEmbedder", lambda backbone: WrongDimEmbedder())

    with pytest.raises(RuntimeError, match="embed_dim=128, expected 512"):
        BaselineDL(dl_cfg=DLBaselineConfig(backbone="resnet18"), device="cpu")


def test_baseline_dl_rejects_embedding_output_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BadForwardEmbedder(nn.Module):
        embed_dim = 512

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 128), dtype=x.dtype, device=x.device)

    image_path = tmp_path / "finger.png"
    ok = cv2.imwrite(str(image_path), np.full((64, 64), 127, dtype=np.uint8))
    assert ok

    monkeypatch.setattr(baseline_dl, "PretrainedEmbedder", lambda backbone: BadForwardEmbedder())
    model = BaselineDL(dl_cfg=DLBaselineConfig(backbone="resnet18", use_mask=False), device="cpu")

    with pytest.raises(RuntimeError, match="has dim=128, expected model.embed_dim=512"):
        model.embed_path(str(image_path), capture="plain")
