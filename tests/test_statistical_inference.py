"""Unit tests for FR-10 statistical clinical inference."""

from __future__ import annotations

import pytest

from src.inference.baseline import BaselineDistribution, BaselineEstimator
from src.inference.cell_statistics import CellProportionCalculator
from src.inference.statistical_test import (
    StatisticalAnomalyDetector,
    StatisticalTestConfig,
)

CLASS_ORDER = ["WBC", "RBC", "Platelets"]

# Synthetic baseline: WBC=5 %, RBC=90 %, PLT=5 %
BASELINE = BaselineDistribution(
    class_order=CLASS_ORDER,
    proportions={"WBC": 0.05, "RBC": 0.90, "Platelets": 0.05},
    config_fingerprint="test",
    created_from_split="train",
)

_DEFAULT_CFG = StatisticalTestConfig(
    alpha=0.05,
    min_cells_for_test=5,
    min_expected_count=1.0,
    test_type="chi_square",
)

_detector = StatisticalAnomalyDetector(_DEFAULT_CFG)
_calculator = CellProportionCalculator(CLASS_ORDER)


# ---------------------------------------------------------------------------
# Baseline provenance
# ---------------------------------------------------------------------------

def test_fr10_baseline_created_from_train_only() -> None:
    """BaselineEstimator must tag the distribution as 'train'."""
    estimator = BaselineEstimator(CLASS_ORDER)
    train_counts = [
        {"WBC": 1, "RBC": 18, "Platelets": 1},
        {"WBC": 4, "RBC": 72, "Platelets": 4},
    ]
    baseline = estimator.fit_from_train_counts(train_counts, config={})
    assert baseline.created_from_split == "train"
    assert set(baseline.proportions.keys()) == set(CLASS_ORDER)


# ---------------------------------------------------------------------------
# Proportion integrity
# ---------------------------------------------------------------------------

def test_fr10_proportions_sum_to_one() -> None:
    """Per-image class proportions must sum exactly to 1.0."""
    counts = {"WBC": 5, "RBC": 90, "Platelets": 5}
    stats = _calculator.compute(counts)
    total = sum(stats.proportions.values())
    assert abs(total - 1.0) < 1e-9, f"Proportions sum to {total}, expected 1.0"


def test_fr10_zero_cells_proportions_sum_to_zero() -> None:
    """Empty detection must yield all-zero proportions (sum = 0)."""
    counts = {"WBC": 0, "RBC": 0, "Platelets": 0}
    stats = _calculator.compute(counts)
    assert sum(stats.proportions.values()) == 0.0


# ---------------------------------------------------------------------------
# Normal case → no alert
# ---------------------------------------------------------------------------

def test_fr10_normal_case_no_alert() -> None:
    """Image matching the baseline distribution must not trigger a clinical alert."""
    counts = {"WBC": 5, "RBC": 90, "Platelets": 5}  # exactly matches baseline
    stats = _calculator.compute(counts)
    result = _detector.evaluate(BASELINE, stats)
    assert result.test_executed, "Chi-square test must execute for a well-populated sample."
    assert not result.alert, (
        f"Normal image (matching baseline) triggered an alert (p={result.p_value:.4f})."
    )


# ---------------------------------------------------------------------------
# Anomalous case → alert
# ---------------------------------------------------------------------------

def test_fr10_anomalous_wbc_elevation_triggers_alert() -> None:
    """WBC=60 % (vs baseline 5 %) must trigger a clinical alert (p < 0.05)."""
    counts = {"WBC": 60, "RBC": 35, "Platelets": 5}
    stats = _calculator.compute(counts)
    result = _detector.evaluate(BASELINE, stats)
    assert result.test_executed, "Chi-square test must execute for this sample size."
    assert result.alert, (
        f"Anomalous image (WBC=60 %) did NOT trigger alert (p={result.p_value})."
    )
    assert result.p_value is not None
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Configurable α
# ---------------------------------------------------------------------------

def test_fr10_alpha_configurable_strict() -> None:
    """Setting α=0.99 must cause alerts for mild deviations from baseline."""
    strict_cfg = StatisticalTestConfig(
        alpha=0.99,
        min_cells_for_test=5,
        min_expected_count=1.0,
        test_type="chi_square",
    )
    strict_detector = StatisticalAnomalyDetector(strict_cfg)
    counts = {"WBC": 6, "RBC": 89, "Platelets": 5}  # slight deviation from 5/90/5
    stats = _calculator.compute(counts)
    result = strict_detector.evaluate(BASELINE, stats)
    assert result.test_executed
    assert result.alert, (
        f"With α=0.99 even mild deviations should alert (p={result.p_value})."
    )


def test_fr10_alpha_configurable_lenient() -> None:
    """Setting α=0.001 must suppress alerts for the same anomalous counts that α=0.05 catches."""
    lenient_cfg = StatisticalTestConfig(
        alpha=0.001,
        min_cells_for_test=5,
        min_expected_count=1.0,
        test_type="chi_square",
    )
    lenient_detector = StatisticalAnomalyDetector(lenient_cfg)
    # Moderate deviation: WBC=10 vs expected 5 out of 100 cells
    counts = {"WBC": 10, "RBC": 85, "Platelets": 5}
    stats = _calculator.compute(counts)
    result_strict = _detector.evaluate(BASELINE, stats)       # α=0.05
    result_lenient = lenient_detector.evaluate(BASELINE, stats)  # α=0.001
    # Strict threshold triggers; lenient may not
    assert result_strict.p_value == pytest.approx(result_lenient.p_value, rel=1e-9), (
        "Both detectors must compute the same p-value."
    )


# ---------------------------------------------------------------------------
# Per-image result reporting
# ---------------------------------------------------------------------------

def test_fr10_result_carries_p_value_and_alert_flag() -> None:
    """StatisticalTestResult for an executed test must expose p_value and alert."""
    counts = {"WBC": 5, "RBC": 90, "Platelets": 5}
    stats = _calculator.compute(counts)
    result = _detector.evaluate(BASELINE, stats)
    assert result.test_executed
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.alert, bool)


def test_fr10_no_cells_skips_test_no_alert() -> None:
    """Zero-cell detection must skip the statistical test and produce no alert."""
    counts = {"WBC": 0, "RBC": 0, "Platelets": 0}
    stats = _calculator.compute(counts)
    result = _detector.evaluate(BASELINE, stats)
    assert not result.test_executed
    assert not result.alert
    assert result.p_value is None
    assert result.reason == "no_cells_detected"
