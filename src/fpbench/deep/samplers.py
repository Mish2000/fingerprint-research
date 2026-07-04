from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sample_weights(labels: Sequence[int | float]) -> torch.DoubleTensor:
    """Return inverse-frequency sample weights for binary labels."""
    labels_int = [int(x) for x in labels]
    if not labels_int:
        raise ValueError("labels must not be empty")
    counts = {0: labels_int.count(0), 1: labels_int.count(1)}
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"balanced sampling requires both classes; counts={counts}")
    weights = [1.0 / float(counts[int(label)]) for label in labels_int]
    return torch.as_tensor(weights, dtype=torch.double)


def build_weighted_random_sampler(
    labels: Sequence[int | float],
    *,
    num_samples: int | None = None,
    replacement: bool = True,
    seed: int | None = None,
) -> WeightedRandomSampler:
    weights = make_balanced_sample_weights(labels)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples) if num_samples is not None else len(weights),
        replacement=bool(replacement),
        generator=generator,
    )
