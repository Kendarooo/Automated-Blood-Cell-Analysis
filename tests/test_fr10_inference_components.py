"""Unit tests for FR-10 inference components."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.baseline import (
    BaselineEstimator,
    BaselineRepository,
    ConfigFingerprint,
)
from src.inference.cell_statistics import (
    CellCountAggregator,
    CellProportionCalculator,
)
from src.inference.statistical_test import (
    StatisticalAnomalyDetector,
    StatisticalTestConfig,
)


CLASS_ORDER = ["WBC", "RBC", "Platelets"]


def test_cell_count_aggregator_counts_labels_in_configured_order() -> None:
    """Predicted labels must be aggregated into configured class counts."""
    aggregator = CellCountAggregator(CLASS_ORDER)
    counts = aggregator.aggregate(["RBC", "RBC", "WBC", "Platelets", "RBC"])

    assert counts == {"WBC": 1, "RBC": 3, "Platelets": 1}


def test_cell_proportions_sum_to_one() -> None:
    """Computed cell proportions must sum to one for non-empty detections."""
    calculator = CellProportionCalculator(CLASS_ORDER)
    statistics = calculator.compute({"WBC": 2, "RBC": 6, "Platelets": 2})

    assert statistics.total_cells == 10
    assert pytest.approx(sum(statistics.proportions.values()), rel=1e-9) == 1.0


def test_cell_proportion_calculator_handles_empty_counts() -> None:
    """Empty detections must not crash and should produce zero proportions."""
    calculator = CellProportionCalculator(CLASS_ORDER)
    statistics = calculator.compute({"WBC": 0, "RBC": 0, "Platelets": 0})

    assert statistics.total_cells == 0
    assert statistics.proportions == {"WBC": 0.0, "RBC": 0.0, "Platelets": 0.0}


def test_baseline_estimator_uses_train_counts_only() -> None:
    """Baseline proportions must be estimated from train counts only."""
    estimator = BaselineEstimator(CLASS_ORDER)
    config = {"inference": {"alpha": 0.05}}
    baseline = estimator.fit_from_train_counts(
        train_counts=[
            {"WBC": 1, "RBC": 8, "Platelets": 1},
            {"WBC": 2, "RBC": 7, "Platelets": 1},
        ],
        config=config,
    )

    assert baseline.created_from_split == "train"
    assert baseline.class_order == CLASS_ORDER
    assert pytest.approx(sum(baseline.proportions.values()), rel=1e-9) == 1.0
    assert baseline.config_fingerprint == ConfigFingerprint.build(config)


def test_baseline_repository_rejects_config_mismatch(tmp_path: Path) -> None:
    """Loading a baseline with a different config must raise an error."""
    estimator = BaselineEstimator(CLASS_ORDER)
    baseline = estimator.fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    path = tmp_path / "baseline.json"
    BaselineRepository.save(baseline, str(path))

    with pytest.raises(ValueError):
        BaselineRepository.load(
            str(path),
            expected_config={"inference": {"alpha": 0.01}},
        )


def test_statistical_detector_flags_no_cells_detected() -> None:
    """An image with no detected cells should not execute the test."""
    detector = StatisticalAnomalyDetector(
        StatisticalTestConfig(
            alpha=0.05,
            min_cells_for_test=5,
            min_expected_count=1.0,
            test_type="chi_square",
        )
    )
    baseline = BaselineEstimator(CLASS_ORDER).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    statistics = CellProportionCalculator(CLASS_ORDER).compute(
        {"WBC": 0, "RBC": 0, "Platelets": 0}
    )

    result = detector.evaluate(baseline, statistics)

    assert not result.test_executed
    assert result.reason == "no_cells_detected"
    assert result.p_value is None


def test_statistical_detector_skips_images_with_too_few_cells() -> None:
    """Very small images should not be overinterpreted statistically."""
    detector = StatisticalAnomalyDetector(
        StatisticalTestConfig(
            alpha=0.05,
            min_cells_for_test=10,
            min_expected_count=1.0,
            test_type="chi_square",
        )
    )
    baseline = BaselineEstimator(CLASS_ORDER).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    statistics = CellProportionCalculator(CLASS_ORDER).compute(
        {"WBC": 1, "RBC": 2, "Platelets": 1}
    )

    result = detector.evaluate(baseline, statistics)

    assert not result.test_executed
    assert result.reason == "too_few_cells_for_test"


def test_statistical_detector_skips_low_expected_count_case() -> None:
    """Chi-square should not run when expected counts are too low."""
    detector = StatisticalAnomalyDetector(
        StatisticalTestConfig(
            alpha=0.05,
            min_cells_for_test=5,
            min_expected_count=2.0,
            test_type="chi_square",
        )
    )
    baseline = BaselineEstimator(CLASS_ORDER).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    statistics = CellProportionCalculator(CLASS_ORDER).compute(
        {"WBC": 1, "RBC": 3, "Platelets": 1}
    )

    result = detector.evaluate(baseline, statistics)

    assert not result.test_executed
    assert result.reason == "expected_counts_too_low"


def test_statistical_detector_runs_and_returns_p_value() -> None:
    """A valid image should produce a p-value and a boolean alert decision."""
    detector = StatisticalAnomalyDetector(
        StatisticalTestConfig(
            alpha=0.05,
            min_cells_for_test=5,
            min_expected_count=1.0,
            test_type="chi_square",
        )
    )
    baseline = BaselineEstimator(CLASS_ORDER).fit_from_train_counts(
        train_counts=[
            {"WBC": 1, "RBC": 8, "Platelets": 1},
            {"WBC": 1, "RBC": 8, "Platelets": 1},
        ],
        config={"inference": {"alpha": 0.05}},
    )
    statistics = CellProportionCalculator(CLASS_ORDER).compute(
        {"WBC": 4, "RBC": 2, "Platelets": 4}
    )

    result = detector.evaluate(baseline, statistics)

    assert result.test_executed
    assert result.reason is None
    assert result.p_value is not None
    assert isinstance(result.alert, bool)


def test_class_order_mismatch_raises_error() -> None:
    """Detector must fail loudly if baseline and statistics use different class order."""
    baseline = BaselineEstimator(CLASS_ORDER).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    statistics = CellProportionCalculator(["RBC", "WBC", "Platelets"]).compute(
        {"RBC": 90, "WBC": 5, "Platelets": 5}
    )
    detector = StatisticalAnomalyDetector(
        StatisticalTestConfig(
            alpha=0.05,
            min_cells_for_test=5,
            min_expected_count=1.0,
            test_type="chi_square",
        )
    )

    with pytest.raises(ValueError, match="class_order mismatch"):
        detector.evaluate(baseline=baseline, statistics=statistics)
