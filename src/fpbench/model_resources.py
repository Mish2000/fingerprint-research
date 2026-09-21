"""Official torchvision weights: explicit preparation and offline-only loading.

These are ImageNet backbones, not fingerprint-trained or pore detection models.
The identifiers preserve the DEFAULT weights used by torchvision 0.17.2.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from src.fpbench.runtime_config import ROOT

WEIGHTS = {
    "resnet18": ("resnet18", "ResNet18_Weights.IMAGENET1K_V1", "resnet18-f37072fd.pth"),
    "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V2", "resnet50-11ad3fa6.pth"),
    "vit_base": ("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1", "vit_b_16-c867db91.pth"),
}


def weights_directory() -> Path:
    return Path(os.getenv("FPBENCH_WEIGHTS_DIR") or ROOT / "artifacts/checkpoints/torchvision")


def _identity(path: Path, backbone: str) -> dict:
    _, identifier, filename = WEIGHTS[backbone]
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if not digest.startswith(filename.rsplit("-", 1)[1].split(".")[0]):
        raise ValueError(f"Official weight checksum mismatch for {backbone}")
    return {"backbone": backbone, "weights": identifier,
            "url": f"https://download.pytorch.org/models/{filename}",
            "filename": filename, "sha256": digest, "size": path.stat().st_size,
            "source": "torchvision / ImageNet", "terms": "https://docs.pytorch.org/vision/stable/models.html"}


def validate_weights(backbone: str) -> tuple[Path, dict]:
    filename = WEIGHTS[backbone][2]
    path = weights_directory() / filename
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Prepared {backbone} weights are missing; run workbench.py prepare-models")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    actual = _identity(path, backbone)
    if recorded != actual:
        raise ValueError(f"Weight manifest mismatch for {backbone}; run explicit preparation")
    return path, actual


def prepare_weights(backbone: str) -> dict:
    """The only download entry point; never called by a model or health route."""
    directory = weights_directory()
    directory.mkdir(parents=True, exist_ok=True)
    filename = WEIGHTS[backbone][2]
    path = directory / filename
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(f"https://download.pytorch.org/models/{filename}", timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                stream.close()
                _identity(temporary, backbone)
                temporary.rename(path)
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)
    identity = _identity(path, backbone)
    manifest = path.with_suffix(".json")
    if manifest.exists():
        validate_weights(backbone)
    else:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2)
    return identity


def load_pretrained_model(backbone: str):
    import torch
    import torchvision.models as tvm

    path, _ = validate_weights(backbone)
    # Construct the architecture without triggering torchvision's downloader,
    # then require a complete, strictly compatible official state dictionary.
    model = getattr(tvm, WEIGHTS[backbone][0])(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model
