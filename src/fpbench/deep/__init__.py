"""Deep learning utilities for fingerprint pair reranking.

Phase 2B introduces ``deep_pair_reranker_v1``: a simple shared-encoder
pair classifier that scores plain-vs-roll fingerprint pairs and emits a
``deep_pair_score`` compatible with the existing plain/roll benchmark
protocol.
"""

from .image_io import load_grayscale_image, resolve_fingerprint_path
from .models import SharedEncoderPairClassifier, build_pair_model
from .pair_dataset import FingerprintPairDataset

__all__ = [
    "FingerprintPairDataset",
    "SharedEncoderPairClassifier",
    "build_pair_model",
    "load_grayscale_image",
    "resolve_fingerprint_path",
]
