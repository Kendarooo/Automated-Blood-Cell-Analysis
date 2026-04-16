"""Tests for formal FR-12 ANN vs SVM comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.compare_classifiers import compare_ann_vs_svm
from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)


def _prepared_features(feature_dim: int = 4) -> PreparedFeatureSplits:
    train_features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.1, 0.2],
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 1.0, 1.2, 1.1],
            [2.0, 2.0, 2.0, 2.0],
            [2.1, 2.0, 2.2, 2.1],
        ],
        dtype=np.float32,
    )[:, :feature_dim]
    train_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    val_features = np.array(
        [
            [0.1, 0.1, 0.0, 0.1],
            [1.05, 1.0, 1.1, 1.05],
            [2.05, 2.0, 2.1, 2.05],
        ],
        dtype=np.float32,
    )[:, :feature_dim]
    val_labels = np.array([0, 1, 2], dtype=np.int64)

    return PreparedFeatureSplits(
        train=FeatureSplit(features=train_features, labels=train_labels),
        val=FeatureSplit(features=val_features, labels=val_labels),
        metadata=FeatureMetadata(
            class_names=("Platelets", "RBC", "WBC"),
            feature_dim=feature_dim,
            truncate_at="layer3",
            projection_dim=None,
            normalization_enabled=True,
            dimensionality_strategy="none",
        ),
    )


def test_compare_ann_vs_svm_uses_same_feature_space_and_returns_side_by_side_metrics(
    tmp_path: Path,
) -> None:
    """FR-12 comparison must report ANN and SVM on the same prepared splits."""
    prepared = _prepared_features()
    merged_cfg = {
        "ann": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
            "epochs": 1,
        },
        "svm": {
            "kernel": "linear",
            "C": 1.0,
            "gamma": "scale",
        },
        "dataset": {"batch_size": 2},
        "device": "cpu",
        "outputs": {"runs_dir": str(tmp_path)},
    }

    comparison = compare_ann_vs_svm(merged_cfg, prepared)

    assert "ann" in comparison
    assert "svm" in comparison
    assert "comparison" in comparison
    assert comparison["comparison"]["same_feature_dim"] is True
    assert comparison["comparison"]["same_class_names"] is True
    assert comparison["comparison"]["same_validation_size"] == 3
    assert comparison["comparison"]["best_by_macro_f1"] in {"ann", "svm", "tie"}
    assert len(comparison["ann"]["val_confusion_matrix"]) == 3
    assert len(comparison["svm"]["val_confusion_matrix"]) == 3
    assert set(comparison["ann"]["val_precision_per_class"].keys()) == {
        "Platelets",
        "RBC",
        "WBC",
    }
    assert set(comparison["svm"]["val_recall_per_class"].keys()) == {
        "Platelets",
        "RBC",
        "WBC",
    }
