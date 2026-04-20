"""Formal ANN vs SVM comparison utilities for FR-12."""

from __future__ import annotations

from typing import Any

from src.experiments.feature_pipeline import PreparedFeatureSplits
from src.experiments.run_ann_experiment import run_ann_experiment
from src.experiments.run_svm_experiment import run_svm_experiment


def compare_ann_vs_svm(
    merged_cfg: dict[str, Any],
    prepared: PreparedFeatureSplits,
) -> dict[str, object]:
    """Compare ANN and SVM on the same prepared validation feature space."""
    ann_result = run_ann_experiment(merged_cfg, prepared)
    svm_result = run_svm_experiment(merged_cfg, prepared)

    ann_metrics = _extract_comparison_metrics(ann_result)
    svm_metrics = _extract_comparison_metrics(svm_result)

    comparison = {
        "same_feature_dim": (
            ann_result["feature_metadata"]["feature_dim"]
            == svm_result["feature_metadata"]["feature_dim"]
        ),
        "same_class_names": (
            ann_result["feature_metadata"]["class_names"]
            == svm_result["feature_metadata"]["class_names"]
        ),
        "same_validation_size": int(prepared.val.labels.shape[0]),
        "best_by_macro_f1": _select_best_classifier(
            ann_macro_f1=float(ann_metrics["val_macro_f1"]),
            svm_macro_f1=float(svm_metrics["val_macro_f1"]),
        ),
    }

    return {
        "ann": ann_metrics,
        "svm": svm_metrics,
        "comparison": comparison,
        "feature_metadata": ann_result["feature_metadata"],
    }


def _extract_comparison_metrics(result: dict[str, object]) -> dict[str, object]:
    """Keep only FR-12-comparison-relevant fields from one runner result."""
    return {
        "val_accuracy": result["val_accuracy"],
        "val_macro_f1": result["val_macro_f1"],
        "val_precision_per_class": result["val_precision_per_class"],
        "val_recall_per_class": result["val_recall_per_class"],
        "val_confusion_matrix": result["val_confusion_matrix"],
        "run_id": result["run_id"],
        "run_summary_path": result["run_summary_path"],
    }


def _select_best_classifier(*, ann_macro_f1: float, svm_macro_f1: float) -> str:
    if ann_macro_f1 > svm_macro_f1:
        return "ann"
    if svm_macro_f1 > ann_macro_f1:
        return "svm"
    return "tie"
