"""Regression tests for persisted SVM runs and their normalizers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)
from src.experiments.run_svm_experiment import run_svm_experiment
from src.features.normalize import FeatureNormalizer
from src.inference.run_loader import load_run


def test_persisted_svm_run_restores_predictions_across_three_classes(
    tmp_path: Path,
) -> None:
    """A saved-and-loaded SVM must still predict all three separable classes."""
    train_raw = np.array(
        [
            [-6.0, -5.5],
            [-5.8, -6.2],
            [-6.3, -5.9],
            [0.0, 0.2],
            [0.2, -0.1],
            [-0.2, 0.1],
            [5.8, 6.1],
            [6.2, 5.9],
            [6.0, 6.3],
        ],
        dtype=np.float32,
    )
    train_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)

    val_raw = np.array(
        [
            [-6.1, -5.8],
            [0.1, 0.0],
            [6.1, 6.0],
        ],
        dtype=np.float32,
    )
    val_labels = np.array([0, 1, 2], dtype=np.int64)

    normalizer = FeatureNormalizer()
    train_features = normalizer.fit_transform(train_raw)
    val_features = normalizer.transform(val_raw)
    prepared = PreparedFeatureSplits(
        train=FeatureSplit(features=train_features, labels=train_labels),
        val=FeatureSplit(features=val_features, labels=val_labels),
        metadata=FeatureMetadata(
            class_names=("Platelets", "RBC", "WBC"),
            feature_dim=2,
            truncate_at="layer3",
            projection_dim=None,
            normalization_enabled=True,
            dimensionality_strategy="none",
        ),
        normalizer=normalizer,
    )
    merged_cfg = {
        "svm": {
            "kernel": "linear",
            "C": 1.0,
            "gamma": "scale",
        },
        "outputs": {"runs_dir": str(tmp_path)},
    }

    result = run_svm_experiment(merged_cfg, prepared)
    loaded = load_run(result["output_dir"])
    loaded_normalizer = FeatureNormalizer.load(
        str(Path(result["output_dir"]) / "normalizer.npz")
    )

    probe_raw = np.array(
        [
            [-6.2, -6.0],
            [0.0, 0.1],
            [6.3, 6.1],
        ],
        dtype=np.float32,
    )
    probe_normalized = loaded_normalizer.transform(probe_raw)
    predictions = loaded.model.predict(probe_normalized)

    assert np.array_equal(np.unique(predictions), np.array([0, 1, 2]))
