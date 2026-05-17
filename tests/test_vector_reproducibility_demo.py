from __future__ import annotations

import numpy as np
import pytest

from scripts.research.run_vector_reproducibility_demo import (
    build_binary_comparison_rows,
    compute_comparison_metrics,
)


def test_identical_vectors_are_exactly_equal_with_zero_difference() -> None:
    vector = np.array([1.0, 0.0, -2.5, 4.25], dtype=np.float32)

    metrics = compute_comparison_metrics(vector, vector.copy(), epsilon=1e-6)

    assert metrics["exact_equal"] is True
    assert metrics["allclose_equal"] is True
    assert metrics["max_abs_diff"] == pytest.approx(0.0)
    assert metrics["mean_abs_diff"] == pytest.approx(0.0)
    assert metrics["nonzero_diff_count"] == 0
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["vector_1_sha256"] == metrics["vector_2_sha256"]
    assert metrics["binary_equal_dimension_count"] == 4
    assert metrics["total_dimensions"] == 4
    assert metrics["binary_equal_dimension_rate"] == pytest.approx(1.0)
    assert metrics["binary_equal_bit_count"] == 128
    assert metrics["total_binary_bit_count"] == 128
    assert metrics["binary_equal_bit_rate"] == pytest.approx(1.0)


def test_tiny_float_difference_is_allclose_but_not_byte_exact() -> None:
    vector_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vector_2 = vector_1.copy()
    vector_2[1] = np.nextafter(vector_2[1], np.float32(3.0), dtype=np.float32)

    metrics = compute_comparison_metrics(vector_1, vector_2, epsilon=1e-6)

    assert metrics["exact_equal"] is False
    assert metrics["allclose_equal"] is True
    assert metrics["max_abs_diff"] > 0.0
    assert metrics["max_abs_diff"] <= 1e-6
    assert metrics["nonzero_diff_count"] == 1
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["vector_1_sha256"] != metrics["vector_2_sha256"]
    assert metrics["binary_equal_dimension_count"] == 2
    assert metrics["total_dimensions"] == 3
    assert metrics["binary_equal_dimension_rate"] == pytest.approx(2 / 3)
    assert metrics["binary_equal_bit_count"] < metrics["total_binary_bit_count"]


def test_binary_comparison_rows_expose_float32_bit_pattern() -> None:
    vector = np.array([1.0, -2.5], dtype=np.float32)

    rows = build_binary_comparison_rows(vector, vector.copy())

    assert rows[0]["float32_hex_run_1"] == "3f800000"
    assert rows[0]["float32_hex_run_2"] == "3f800000"
    assert rows[0]["float32_binary_run_1"] == "00111111100000000000000000000000"
    assert rows[0]["float32_binary_run_2"] == "00111111100000000000000000000000"
    assert rows[0]["binary_equal"] is True
    assert len(rows[0]["float32_binary_run_1"]) == 32
