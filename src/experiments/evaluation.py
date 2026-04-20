"""Shared multiclass evaluation utilities for FR-11 sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Comparable validation metrics for ANN and SVM sweeps."""

    accuracy: float
    macro_f1: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    confusion_matrix: np.ndarray


class MulticlassEvaluator:
    """Calculate stable multiclass metrics with explicit class ordering."""

    def __init__(self, class_names: tuple[str, ...]) -> None:
        if not class_names:
            raise ValueError("class_names must contain at least one class.")
        self._class_names = class_names
        self._labels = list(range(len(class_names)))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ClassificationMetrics:
        """Evaluate predictions with zero-division-safe per-class metrics."""
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        precision_values = precision_score(
            y_true,
            y_pred,
            labels=self._labels,
            average=None,
            zero_division=0,
        )
        recall_values = recall_score(
            y_true,
            y_pred,
            labels=self._labels,
            average=None,
            zero_division=0,
        )

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=self._labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            precision_per_class={
                class_name: float(value)
                for class_name, value in zip(self._class_names, precision_values)
            },
            recall_per_class={
                class_name: float(value)
                for class_name, value in zip(self._class_names, recall_values)
            },
            confusion_matrix=confusion_matrix(
                y_true,
                y_pred,
                labels=self._labels,
            ),
        )
