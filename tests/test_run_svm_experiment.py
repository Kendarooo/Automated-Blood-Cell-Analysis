"""Tests for the SVM experiment runner used in FR-11."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)
from src.experiments import run_svm_experiment as svm_runner_module
from src.experiments.run_svm_experiment import run_svm_experiment


def _prepared_features(feature_dim: int = 2) -> PreparedFeatureSplits:
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


def test_run_svm_experiment_returns_serializable_metrics() -> None:
    """The SVM runner must expose selection info, metrics, and feature metadata."""
    prepared = _prepared_features()
    merged_cfg = {
        "svm": {
            "c_values": [0.1, 1.0],
            "gamma_values": ["scale", 0.5],
        }
    }

    result = run_svm_experiment(merged_cfg, prepared)

    assert "val_macro_f1" in result
    assert "val_confusion_matrix" in result
    assert "feature_metadata" in result
    assert "selected_kernel" in result
    assert "selected_c_value" in result
    assert "selected_gamma" in result
    assert "num_evaluated_configs" in result
    assert not np.isnan(result["val_macro_f1"])
    assert len(result["val_confusion_matrix"]) == 3
    assert len(result["val_confusion_matrix"][0]) == 3
    assert result["feature_metadata"]["feature_dim"] == prepared.metadata.feature_dim
    assert result["num_evaluated_configs"] == 6
    assert "run_id" in result
    assert "output_dir" in result
    assert Path(result["run_summary_path"]).exists()
    assert Path(result["local_artifact_paths"]["model"]).exists()


def test_run_svm_experiment_uses_prepared_feature_dim_and_selects_model() -> None:
    """The SVM runner must consume the prepared feature space without rebuilding it."""
    prepared = _prepared_features(feature_dim=2)
    merged_cfg = {
        "svm": {
            "c_values": [1.0],
            "gamma_values": ["scale"],
        }
    }

    result = run_svm_experiment(merged_cfg, prepared)

    assert result["feature_metadata"]["feature_dim"] == 2
    assert result["selected_kernel"] in {"linear", "rbf"}
    assert 0.0 <= result["val_accuracy"] <= 1.0


def test_run_svm_experiment_accepts_point_config_for_sweeps() -> None:
    """Point-config mode must evaluate a single SVM configuration."""
    prepared = _prepared_features(feature_dim=2)
    merged_cfg = {
        "svm": {
            "kernel": "linear",
            "C": 1.0,
            "gamma": "scale",
        }
    }

    result = run_svm_experiment(merged_cfg, prepared)

    assert result["selected_kernel"] == "linear"
    assert result["selected_c_value"] == 1.0
    assert result["selected_gamma"] == "scale"
    assert result["num_evaluated_configs"] == 1


def test_run_svm_experiment_logs_to_wandb_when_enabled(monkeypatch) -> None:
    """When enabled, the SVM runner must init, log, and finish a W&B run."""
    prepared = _prepared_features()
    merged_cfg = {
        "wandb": {
            "enabled": True,
            "project": "bccd-hemograma",
            "entity": "team",
            "run_name": "svm-test-run",
        },
        "svm": {
            "c_values": [0.1, 1.0],
            "gamma_values": ["scale", 0.5],
        },
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

    monkeypatch.setattr(svm_runner_module.wandb, "init", fake_init)
    monkeypatch.setattr(svm_runner_module.wandb, "log", fake_log)
    monkeypatch.setattr(svm_runner_module.wandb, "finish", fake_finish)

    result = run_svm_experiment(merged_cfg, prepared)

    assert result["val_macro_f1"] >= 0.0
    assert captured["init"] is not None
    assert captured["init"]["project"] == "bccd-hemograma"
    assert captured["init"]["entity"] == "team"
    assert captured["init"]["name"] == "svm-test-run"
    assert captured["log"] is not None
    assert "val_macro_f1" in captured["log"]
    assert "val_accuracy" in captured["log"]
    assert "val_precision_Platelets" in captured["log"]
    assert "val_recall_RBC" in captured["log"]
    assert "selected_kernel" in captured["log"]
    assert captured["finish_calls"] == 1
