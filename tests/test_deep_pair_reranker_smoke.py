from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.fpbench.deep.models import build_pair_model
from src.fpbench.deep.samplers import make_balanced_sample_weights


def test_small_pair_model_forward_shape() -> None:
    model = build_pair_model(backbone="small_cnn", channels=1, embedding_dim=32, hidden_dim=64)
    image_a = torch.randn(2, 1, 64, 64)
    image_b = torch.randn(2, 1, 64, 64)
    logits = model(image_a, image_b)
    assert tuple(logits.shape) == (2,)
    assert torch.isfinite(logits).all()


def test_balanced_weights_require_both_classes() -> None:
    weights = make_balanced_sample_weights([1, 1, 0, 0, 0, 0])
    assert len(weights) == 6
    assert weights[0] > weights[-1]
