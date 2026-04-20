"""Tests for persisted run restoration in FR-13."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)
from src.experiments.run_ann_experiment import run_ann_experiment
from src.experiments.run_svm_experiment import run_svm_experiment
from src.inference.run_loader import load_run


def _prepared_ann_features(feature_dim: int = 4) -> PreparedFeatureSplits:
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


def _prepared_svm_features(feature_dim: int = 2) -> PreparedFeatureSplits:
    train_features = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 1.2],
        [2.0, 2.0],
        [2.1, 2.2],
    ], dtype=np.float32)[:, :feature_dim]
    train_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    val_features = np.array([
        [0.05, 0.1],
        [1.05, 1.1],
        [2.05, 2.1],
    ], dtype=np.float32)[:, :feature_dim]
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


def test_load_run_restores_ann_model_without_retraining(tmp_path: Path) -> None:
    """ANN persisted runs must restore architecture, weights, and metadata."""
    prepared = _prepared_ann_features()
    merged_cfg = {
        "ann": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
            "epochs": 1,
        },
        "dataset": {"batch_size": 2},
        "device": "cpu",
        "outputs": {"runs_dir": str(tmp_path)},
    }

    result = run_ann_experiment(merged_cfg, prepared)
    loaded = load_run(result["output_dir"])

    assert loaded.classifier_name == "ann"
    assert loaded.feature_metadata["feature_dim"] == prepared.metadata.feature_dim
    assert loaded.effective_config["ann"]["hidden_dims"] == [8]
    assert isinstance(loaded.model, torch.nn.Module)
    assert loaded.model.training is False
    logits = loaded.model(torch.tensor(prepared.val.features, dtype=torch.float32))
    assert logits.shape == (3, 3)


def test_load_run_restores_svm_model_without_retraining(tmp_path: Path) -> None:
    """SVM persisted runs must restore the trained classifier directly."""
    prepared = _prepared_svm_features()
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

    assert loaded.classifier_name == "svm"
    assert loaded.feature_metadata["feature_dim"] == prepared.metadata.feature_dim
    predictions = loaded.model.predict(prepared.val.features)
    assert predictions.shape == (3,)


def test_load_run_rejects_unknown_classifier(tmp_path: Path) -> None:
    """Unknown classifier names in the persisted summary must fail clearly."""
    run_dir = tmp_path / "mystery_run"
    run_dir.mkdir(parents=True)
    (run_dir / "effective_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"feature_dim": 2}),
        encoding="utf-8",
    )
    model_path = run_dir / "model.bin"
    model_path.write_text("x", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "classifier_name": "knn",
                "local_artifact_paths": {"model": str(model_path)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported classifier_name"):
        load_run(run_dir)


def test_load_run_requires_summary_file(tmp_path: Path) -> None:
    """Missing persisted summary must fail before any partial restoration."""
    with pytest.raises(FileNotFoundError, match="run_summary.json"):
        load_run(tmp_path / "missing_run")
