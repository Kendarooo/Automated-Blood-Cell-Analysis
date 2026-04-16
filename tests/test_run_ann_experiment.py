"""Tests for the ANN experiment runner used in FR-11."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)
from src.experiments import run_ann_experiment as ann_runner_module
from src.experiments.run_ann_experiment import run_ann_experiment


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
            class_names=("WBC", "RBC", "Platelets"),
            feature_dim=feature_dim,
            truncate_at="layer3",
            projection_dim=None,
            normalization_enabled=True,
            dimensionality_strategy="none",
        ),
    )


def test_run_ann_experiment_returns_serializable_metrics() -> None:
    """The ANN runner must expose validation metrics and feature metadata."""
    prepared = _prepared_features()
    merged_cfg = {
        "ann": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
            "epochs": 2,
        },
        "dataset": {
            "batch_size": 2,
        },
        "device": "cpu",
    }

    result = run_ann_experiment(merged_cfg, prepared)

    assert "val_macro_f1" in result
    assert "val_confusion_matrix" in result
    assert "feature_metadata" in result
    assert result["batch_size"] == 2
    assert result["epochs"] == 2
    assert not np.isnan(result["val_macro_f1"])
    assert len(result["val_confusion_matrix"]) == 3
    assert len(result["val_confusion_matrix"][0]) == 3
    assert result["feature_metadata"]["feature_dim"] == prepared.metadata.feature_dim
    assert "run_id" in result
    assert "output_dir" in result
    assert Path(result["run_summary_path"]).exists()
    assert Path(result["local_artifact_paths"]["model"]).exists()


def test_run_ann_experiment_uses_feature_dim_from_prepared_splits() -> None:
    """The ANN input dimensionality must come from PreparedFeatureSplits."""
    prepared = _prepared_features(feature_dim=3)
    merged_cfg = {
        "ann": {
            "hidden_dims": [6],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
            "epochs": 1,
        },
        "dataset": {
            "batch_size": 3,
        },
        "device": "cpu",
    }

    result = run_ann_experiment(merged_cfg, prepared)

    assert result["feature_metadata"]["feature_dim"] == 3
    assert 0.0 <= result["val_accuracy"] <= 1.0


def test_run_ann_experiment_logs_to_wandb_when_enabled(monkeypatch) -> None:
    """When enabled, the ANN runner must init, log, and finish a W&B run."""
    prepared = _prepared_features()
    merged_cfg = {
        "wandb": {
            "enabled": True,
            "project": "bccd-hemograma",
            "entity": "team",
            "run_name": "ann-test-run",
        },
        "ann": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
            "epochs": 1,
        },
        "dataset": {
            "batch_size": 2,
        },
        "device": "cpu",
    }

    captured: dict[str, object] = {
        "init": None,
        "log": None,
        "finish_calls": 0,
    }

    def fake_init(**kwargs):
        captured["init"] = kwargs
        return SimpleNamespace()

    def fake_log(payload):
        captured["log"] = payload

    def fake_finish():
        captured["finish_calls"] += 1

    monkeypatch.setattr(ann_runner_module.wandb, "init", fake_init)
    monkeypatch.setattr(ann_runner_module.wandb, "log", fake_log)
    monkeypatch.setattr(ann_runner_module.wandb, "finish", fake_finish)

    result = run_ann_experiment(merged_cfg, prepared)

    assert result["val_macro_f1"] >= 0.0
    assert captured["init"] is not None
    assert captured["init"]["project"] == "bccd-hemograma"
    assert captured["init"]["entity"] == "team"
    assert captured["init"]["name"] == "ann-test-run"
    assert captured["log"] is not None
    assert "val_macro_f1" in captured["log"]
    assert "val_accuracy" in captured["log"]
    assert "val_precision_WBC" in captured["log"]
    assert "val_recall_RBC" in captured["log"]
    assert captured["finish_calls"] == 1
