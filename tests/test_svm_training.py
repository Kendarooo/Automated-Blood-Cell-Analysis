"""Unit tests for SVM training and validation comparison (FR-9)."""

import numpy as np

from src.training.train_svm import DatasetSplit, SVMTrainer


def test_svm_trainer_compares_linear_and_rbf_on_validation() -> None:
    """
    FR-9:
    train on the training split, compare linear vs. RBF on validation,
    and select the best configuration using validation accuracy.
    """
    train_features = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 1.2],
        [2.0, 2.0],
        [2.1, 2.2],
    ])
    train_labels = np.array([0, 0, 1, 1, 2, 2])

    val_features = np.array([
        [0.05, 0.1],
        [1.05, 1.1],
        [2.05, 2.1],
    ])
    val_labels = np.array([0, 1, 2])

    trainer = SVMTrainer()
    results = trainer.compare_kernels(
        train_split=DatasetSplit(features=train_features, labels=train_labels),
        val_split=DatasetSplit(features=val_features, labels=val_labels),
        c_values=[0.1, 1.0],
        gamma_values=["scale", 0.5],
    )

    assert results, "Expected at least one evaluated SVM configuration."
    assert any(result.kernel == "linear" for result in results)
    assert any(result.kernel == "rbf" for result in results)

    best = trainer.select_best(results)
    assert best.kernel in {"linear", "rbf"}
    assert 0.0 <= best.accuracy <= 1.0
