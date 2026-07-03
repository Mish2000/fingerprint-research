from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class SmallFingerprintEncoder(nn.Module):
    """Small CNN encoder for the first measurable prototype.

    It is intentionally lightweight and does not require downloading external
    pretrained weights. This keeps tests and local smoke runs reliable.
    """

    def __init__(self, *, in_channels: int = 1, embedding_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.features(x)), dim=1)


class ResNet18Encoder(nn.Module):
    def __init__(self, *, in_channels: int = 1, embedding_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - depends on optional torchvision install
            raise RuntimeError("torchvision is required for backbone='resnet18'") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        if in_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained and old.weight.shape[1] == 3:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        in_features = int(base.fc.in_features)
        base.fc = nn.Identity()
        self.base = base
        self.projection = nn.Sequential(nn.Linear(in_features, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.base(x)), dim=1)


class SharedEncoderPairClassifier(nn.Module):
    """Shared-encoder pair classifier.

    Pair feature vector: ``[emb_a, emb_b, abs(emb_a-emb_b), emb_a*emb_b]``.
    """

    def __init__(self, encoder: nn.Module, *, embedding_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = int(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(int(embedding_dim) * 4, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.encode(image_a)
        emb_b = self.encode(image_b)
        pair_features = torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=1)
        return self.head(pair_features).squeeze(1)


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "small_cnn"
    channels: int = 1
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.15
    pretrained: bool = False


def build_pair_model(
    *,
    backbone: Literal["small_cnn", "resnet18"] | str = "small_cnn",
    channels: int = 1,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.15,
    pretrained: bool = False,
) -> SharedEncoderPairClassifier:
    if backbone == "small_cnn":
        encoder = SmallFingerprintEncoder(in_channels=int(channels), embedding_dim=int(embedding_dim))
    elif backbone == "resnet18":
        encoder = ResNet18Encoder(in_channels=int(channels), embedding_dim=int(embedding_dim), pretrained=bool(pretrained))
    else:
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    return SharedEncoderPairClassifier(
        encoder,
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
