"""Cell counting and proportion utilities for FR-10 statistical inference."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CellStatistics:
    """Counts and proportions for one inferred blood-smear image."""

    counts: dict[str, int]
    proportions: dict[str, float]
    total_cells: int


class CellCountAggregator:
    """Aggregate predicted class labels into class counts."""

    def __init__(self, class_order: list[str]) -> None:
        self._class_order = class_order

    def aggregate(self, labels: list[str]) -> dict[str, int]:
        """
        Count the number of predicted cells per class.

        Args:
            labels: Predicted cell labels for one image.

        Returns:
            Dictionary with one count per known class.
        """
        raw_counts = Counter(labels)
        return {
            class_name: int(raw_counts.get(class_name, 0))
            for class_name in self._class_order
        }


class CellProportionCalculator:
    """Convert counts into proportions while handling empty detections safely."""

    def __init__(self, class_order: list[str]) -> None:
        self._class_order = class_order

    def compute(self, counts: dict[str, int]) -> CellStatistics:
        """
        Compute per-class proportions from counts.

        Args:
            counts: Dictionary mapping class names to detected counts.

        Returns:
            CellStatistics containing counts, proportions, and total count.
        """
        total_cells = sum(counts.get(class_name, 0) for class_name in self._class_order)

        normalized_counts = {
            class_name: int(counts.get(class_name, 0))
            for class_name in self._class_order
        }

        if total_cells == 0:
            proportions = {class_name: 0.0 for class_name in self._class_order}
            return CellStatistics(
                counts=normalized_counts,
                proportions=proportions,
                total_cells=0,
            )

        proportions = {
            class_name: normalized_counts[class_name] / total_cells
            for class_name in self._class_order
        }
        return CellStatistics(
            counts=normalized_counts,
            proportions=proportions,
            total_cells=total_cells,
        )
