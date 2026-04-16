"""Tests for shared multiclass evaluation in FR-11 sweeps."""

from __future__ import annotations

import numpy as np

from src.experiments.evaluation import MulticlassEvaluator


CLASS_NAMES = ("WBC", "RBC", "Platelets")


def test_evaluator_returns_confusion_matrix_with_expected_shape() -> None:
    """Confusion matrix must use the configured multiclass order."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics.confusion_matrix.shape == (3, 3)


def test_evaluator_respects_configured_class_order() -> None:
    """Per-class dictionaries must preserve the configured class ordering."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert list(metrics.precision_per_class.keys()) == list(CLASS_NAMES)
    assert list(metrics.recall_per_class.keys()) == list(CLASS_NAMES)


def test_evaluator_computes_macro_f1_for_simple_case() -> None:
    """A simple perfect prediction case should produce macro F1 of 1.0."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0


def test_evaluator_reports_precision_and_recall_per_class() -> None:
    """Per-class precision and recall must match a known confusion pattern."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics.precision_per_class["WBC"] == 0.5
    assert metrics.recall_per_class["WBC"] == 0.5
    assert metrics.precision_per_class["RBC"] == 2 / 3
    assert metrics.recall_per_class["RBC"] == 1.0


def test_evaluator_handles_class_never_predicted() -> None:
    """A missing predicted class must yield zero precision/recall without NaNs."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([1, 1, 1, 1, 2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics.precision_per_class["WBC"] == 0.0
    assert metrics.recall_per_class["WBC"] == 0.0
    assert not np.isnan(metrics.macro_f1)


def test_evaluator_handles_class_absent_in_ground_truth() -> None:
    """Classes absent in y_true must not crash and should remain finite."""
    evaluator = MulticlassEvaluator(CLASS_NAMES)
    y_true = np.array([1, 1, 1, 2, 2])
    y_pred = np.array([1, 0, 1, 2, 2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics.recall_per_class["WBC"] == 0.0
    assert not np.isnan(metrics.macro_f1)
