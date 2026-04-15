"""Statistical anomaly detection for FR-10 clinical inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chisquare

from src.inference.baseline import BaselineDistribution
from src.inference.cell_statistics import CellStatistics


@dataclass(frozen=True)
class StatisticalTestConfig:
    """Configuration for the anomaly detection statistical test."""

    alpha: float
    min_cells_for_test: int
    min_expected_count: float
    test_type: str


@dataclass(frozen=True)
class StatisticalTestResult:
    """Result of one anomaly detection decision."""

    p_value: float | None
    alert: bool
    test_executed: bool
    reason: str | None


class ExpectedCountValidator:
    """
    Validate whether a chi-square approximation is trustworthy.

    The chi-square goodness-of-fit approximation becomes unreliable when
    expected counts are too small, which is especially relevant for rare
    cell classes in small images.
    """

    def __init__(self, min_expected_count: float) -> None:
        self._min_expected_count = min_expected_count

    def validate(self, expected_counts: np.ndarray) -> tuple[bool, str | None]:
        """
        Check whether every expected count is above the configured threshold.

        Returns:
            Tuple (is_valid, reason_if_invalid).
        """
        if np.any(expected_counts < self._min_expected_count):
            return False, "expected_counts_too_low"
        return True, None


class StatisticalAnomalyDetector:
    """Run statistical anomaly detection against a train-derived baseline."""

    def __init__(self, config: StatisticalTestConfig) -> None:
        self._config = config
        self._expected_count_validator = ExpectedCountValidator(
            min_expected_count=config.min_expected_count,
        )

    def evaluate(
        self,
        baseline: BaselineDistribution,
        statistics: CellStatistics,
    ) -> StatisticalTestResult:
        """
        Compare observed counts against the baseline distribution.

        Args:
            baseline: Baseline estimated from training data only.
            statistics: Counts/proportions computed for one inferred image.

        Returns:
            StatisticalTestResult with p-value and alert decision.
        """
        if list(baseline.class_order) != list(statistics.counts.keys()):
            raise ValueError(
                f"class_order mismatch: baseline={baseline.class_order}, "
                f"statistics={list(statistics.counts.keys())}"
            )

        if statistics.total_cells == 0:
            return StatisticalTestResult(
                p_value=None,
                alert=False,
                test_executed=False,
                reason="no_cells_detected",
            )

        if statistics.total_cells < self._config.min_cells_for_test:
            return StatisticalTestResult(
                p_value=None,
                alert=False,
                test_executed=False,
                reason="too_few_cells_for_test",
            )

        observed_counts = np.array(
            [statistics.counts[class_name] for class_name in baseline.class_order],
            dtype=float,
        )
        expected_counts = np.array(
            [
                statistics.total_cells * baseline.proportions[class_name]
                for class_name in baseline.class_order
            ],
            dtype=float,
        )

        is_valid, reason = self._expected_count_validator.validate(expected_counts)
        if not is_valid:
            return StatisticalTestResult(
                p_value=None,
                alert=False,
                test_executed=False,
                reason=reason,
            )

        if self._config.test_type != "chi_square":
            raise ValueError(
                f"Unsupported statistical test '{self._config.test_type}'."
            )

        _, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
        return StatisticalTestResult(
            p_value=float(p_value),
            alert=bool(p_value < self._config.alpha),
            test_executed=True,
            reason=None,
        )
