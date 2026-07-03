from __future__ import annotations

import numpy as np


def binary_confusion_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels = labels[valid]
    scores = scores[valid]
    accepted = scores >= float(threshold)
    pos = labels == 1
    neg = labels == 0
    ta = int(np.sum(accepted & pos))
    fr = int(np.sum((~accepted) & pos))
    fa = int(np.sum(accepted & neg))
    tr = int(np.sum((~accepted) & neg))
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    return {
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "tar": float(ta / n_pos) if n_pos else float("nan"),
        "far": float(fa / n_neg) if n_neg else float("nan"),
    }
