from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import cv2
import numpy as np
# Torch / torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fpbench.preprocess.preprocess import (
    PreprocessConfig,
    extract_fingerprint_roi,
    load_gray,
    preprocess_image,
    suppress_header_and_borders,
    rectangular_gate_mask,
)

BackboneName = Literal["resnet18", "resnet50", "vit_base"]


@dataclass
class DLBaselineConfig:
    backbone: BackboneName = "resnet50"
    input_size: int = 224               # ResNet default
    use_mask: bool = True               # apply ROI/gate masking
    roi_min_frac: float = 0.02          # ROI sanity
    roi_max_frac: float = 0.85
    gate_top_plain: float = 0.18        # tuned in Week 2 philosophy
    gate_top_roll: float = 0.05
    gate_border: int = 12


def _infer_capture_from_path(path: str) -> Optional[str]:
    s = path.lower()
    if "contactless" in s or "contact-less" in s or "contact_less" in s:
        return "contactless"
    if "contact_based" in s or "contact-based" in s or "contactbased" in s:
        return "contact_based"
    if "plain" in s:
        return "plain"
    if "roll" in s or "rolled" in s:
        return "roll"
    return None


def _normalize_capture(capture: Optional[str], *, path: Optional[str] = None) -> str:
    aliases = {
        "plain": "plain",
        "roll": "roll",
        "rolled": "roll",
        "contactless": "contactless",
        "contact-less": "contactless",
        "contact_less": "contactless",
        "contact_based": "contact_based",
        "contact-based": "contact_based",
        "contactbased": "contact_based",
    }
    raw = str(capture or "").strip().lower()
    if raw:
        if raw not in aliases:
            raise ValueError(f"Unsupported capture label: {capture}")
        return aliases[raw]
    inferred = _infer_capture_from_path(path or "")
    if inferred:
        return inferred
    return "plain"


def _parse_file_uri(p: str) -> str:
    # Supports: file:/C:/... or normal path
    if p.startswith("file:"):
        p = p[len("file:"):]
        if p.startswith("/"):
            p = p[1:]
    return p


def _build_final_mask(img_u8: np.ndarray, cfg: DLBaselineConfig, capture: Optional[str], *, path: Optional[str] = None) -> np.ndarray:
    """
    Build final mask (0/255 uint8) with ROI+gate policy + fallback.
    Explicit capture is primary. Path inference is last-resort only.
    """
    h, w = img_u8.shape[:2]
    cap_norm = _normalize_capture(capture, path=path)
    gate_capture = "plain" if cap_norm in {"plain", "contactless"} else "roll"

    gate = rectangular_gate_mask(
        (h, w),
        capture=gate_capture,
        top_plain=cfg.gate_top_plain,
        top_roll=cfg.gate_top_roll,
        border=cfg.gate_border,
    )

    roi_result = extract_fingerprint_roi(img_u8)
    roi = suppress_header_and_borders(roi_result.mask, top_ratio=0.12, border=10)

    if roi_result.is_valid:
        final = cv2.bitwise_and(roi, gate)
        frac = float((final > 0).mean())
        # If ROI is unreasonable, fall back to deterministic gate explicitly.
        if frac < cfg.roi_min_frac or frac > cfg.roi_max_frac:
            final = gate
    else:
        final = gate

    return final


class PretrainedEmbedder(nn.Module):
    def __init__(self, backbone: BackboneName = "resnet50"):
        super().__init__()
        try:
            import torchvision.models as tvm
            if backbone == "resnet18":
                m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
                dim = 512
                self.backbone = nn.Sequential(*list(m.children())[:-1])
                self.is_vit = False
            elif backbone == "resnet50":
                m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
                dim = 2048
                self.backbone = nn.Sequential(*list(m.children())[:-1])
                self.is_vit = False
            elif backbone == "vit_base":
                m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
                m.heads = nn.Identity()
                dim = 768
                self.backbone = m
                self.is_vit = True
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        except Exception:
            dim = 128
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(64, dim),
            )
            self.is_vit = False

        self.embed_dim = dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vit:
            z = self.backbone(x)                 # (B, 768)
        else:
            z = self.backbone(x)                 # (B, dim, 1, 1)
            z = z.flatten(1)                     # (B, dim)
        z = F.normalize(z, p=2, dim=1)           # L2 normalize
        return z


class BaselineDL:
    def __init__(
        self,
        dl_cfg: DLBaselineConfig | None = None,
        prep_cfg: PreprocessConfig | None = None,
        device: str | None = None,
    ):
        self.dl_cfg = dl_cfg or DLBaselineConfig()
        self.prep_cfg = prep_cfg or PreprocessConfig(target_size=512)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = PretrainedEmbedder(self.dl_cfg.backbone).to(self.device).eval()

        # ImageNet normalization (torchvision convention)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def config_dict(self) -> dict:
        return {
            "dl_cfg": asdict(self.dl_cfg),
            "prep_cfg": asdict(self.prep_cfg),
            "device": self.device,
            "embed_dim": int(self.model.embed_dim),
        }

    def _image_to_tensor(self, img_u8: np.ndarray) -> torch.Tensor:
        # img_u8: HxW uint8 in [0,255]
        x = torch.from_numpy(img_u8).to(self.device, dtype=torch.float32) / 255.0  # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)                                            # (1,1,H,W)
        x = x.repeat(1, 3, 1, 1)                                                   # (1,3,H,W)
        x = F.interpolate(x, size=(self.dl_cfg.input_size, self.dl_cfg.input_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def embed_path(self, path_str: str, capture: Optional[str] = None) -> tuple[np.ndarray, float]:
        """
        Returns (embedding float32 [D], embed_ms).
        """
        t0 = time.perf_counter()

        p = _parse_file_uri(path_str)
        gray = load_gray(p)
        img = preprocess_image(gray, self.prep_cfg)

        if self.dl_cfg.use_mask:
            m = _build_final_mask(img, self.dl_cfg, capture, path=p)
            img = cv2.bitwise_and(img, m)

        x = self._image_to_tensor(img)
        z = self.model(x)[0].detach().to("cpu").numpy().astype(np.float32)

        ms = (time.perf_counter() - t0) * 1000.0
        return z, ms

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        # embeddings are already L2-normalized
        return float(np.dot(a, b))
