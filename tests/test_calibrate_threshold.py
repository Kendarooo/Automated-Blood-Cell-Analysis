"""Tests for confidence-threshold calibration helpers."""

from __future__ import annotations

import pytest

from src.evaluation.calibrate_threshold import compute_detection_metrics


def test_compute_detection_metrics_returns_expected_precision_recall_f1() -> None:
    """Precision/recall/F1 should follow the standard matched/FP/FN formulas."""
    metrics = compute_detection_metrics(
        matched=8,
        false_positive=2,
        false_negative=4,
    )

    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(8 / 12)
    assert metrics["f1"] == pytest.approx(0.7272727272727272)


def test_compute_detection_metrics_handles_zero_denominators() -> None:
    """Empty counts should not raise and should return zeros."""
    metrics = compute_detection_metrics(
        matched=0,
        false_positive=0,
        false_negative=0,
    )

    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
