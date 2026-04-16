"""Unit tests for artifact loading in FR-10 inference."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.baseline import BaselineEstimator, BaselineRepository
from src.inference.artifact_loader import ArtifactLoader, ArtifactPaths
from src.experiments.feature_pipeline import (
    FeatureMetadata,
    FeatureSplit,
    PreparedFeatureSplits,
)
from src.experiments.run_ann_experiment import run_ann_experiment
from src.experiments.run_svm_experiment import run_svm_experiment


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


def test_artifact_loader_fails_clearly_if_path_missing() -> None:
    """Missing artifact paths must raise a clear FileNotFoundError."""
    paths = ArtifactPaths(
        detector=Path("no_existe.pt"),
        extractor_config=Path("extractor_config.json"),
        classifier=Path("classifier.joblib"),
        normalizer=Path("normalizer.npz"),
        baseline=Path("baseline.json"),
    )

    with pytest.raises(FileNotFoundError, match="no_existe.pt"):
        ArtifactLoader().load(paths, config={})


def test_artifact_loader_rejects_baseline_with_config_mismatch(tmp_path: Path) -> None:
    """Loader must fail if the saved baseline does not match the current config."""
    detector_path = tmp_path / "detector.pt"
    extractor_config_path = tmp_path / "extractor_config.json"
    classifier_path = tmp_path / "classifier.joblib"
    normalizer_path = tmp_path / "normalizer.npz"
    baseline_path = tmp_path / "baseline.json"

    detector_path.write_text("detector", encoding="utf-8")
    extractor_config_path.write_text("{}", encoding="utf-8")
    classifier_path.write_text("classifier", encoding="utf-8")
    normalizer_path.write_text("normalizer", encoding="utf-8")

    baseline = BaselineEstimator(["Platelets", "RBC", "WBC"]).fit_from_train_counts(
        train_counts=[{"Platelets": 1, "RBC": 8, "WBC": 1}],
        config={"inference": {"alpha": 0.05}},
    )
    BaselineRepository.save(baseline, str(baseline_path))

    paths = ArtifactPaths(
        detector=detector_path,
        extractor_config=extractor_config_path,
        classifier=classifier_path,
        normalizer=normalizer_path,
        baseline=baseline_path,
    )

    with pytest.raises(ValueError, match="configuration"):
        ArtifactLoader().load(paths, config={"inference": {"alpha": 0.01}})


def test_artifact_loader_rejects_unexpected_classifier_type(tmp_path: Path) -> None:
    """Loader must reject classifier artifacts that do not match the expected type."""
    detector_path = tmp_path / "detector.pt"
    extractor_config_path = tmp_path / "extractor_config.json"
    classifier_path = tmp_path / "classifier.joblib"
    normalizer_path = tmp_path / "normalizer.npz"
    baseline_path = tmp_path / "baseline.json"

    detector_path.write_text("detector", encoding="utf-8")
    extractor_config_path.write_text("{}", encoding="utf-8")
    classifier_path.write_text("classifier", encoding="utf-8")
    normalizer_path.write_text("normalizer", encoding="utf-8")

    baseline = BaselineEstimator(["Platelets", "RBC", "WBC"]).fit_from_train_counts(
        train_counts=[{"Platelets": 1, "RBC": 8, "WBC": 1}],
        config={"classifier": {"type": "ann"}},
    )
    BaselineRepository.save(baseline, str(baseline_path))

    paths = ArtifactPaths(
        detector=detector_path,
        extractor_config=extractor_config_path,
        classifier=classifier_path,
        normalizer=normalizer_path,
        baseline=baseline_path,
    )

    with pytest.raises(ValueError, match="classifier type"):
        ArtifactLoader().load(paths, config={"classifier": {"type": "ann"}})


def test_artifact_loader_restores_ann_from_run_dir(tmp_path: Path) -> None:
    """ArtifactLoader must restore an ANN classifier from a persisted run dir."""
    detector_path = tmp_path / "detector.pt"
    extractor_config_path = tmp_path / "extractor_config.json"
    normalizer_path = tmp_path / "normalizer.npz"
    baseline_path = tmp_path / "baseline.json"

    detector_path.write_text("detector", encoding="utf-8")
    extractor_config_path.write_text("{}", encoding="utf-8")
    normalizer_path.write_text("normalizer", encoding="utf-8")

    baseline = BaselineEstimator(["Platelets", "RBC", "WBC"]).fit_from_train_counts(
        train_counts=[{"Platelets": 1, "RBC": 8, "WBC": 1}],
        config={"classifier": {"type": "ann"}},
    )
    BaselineRepository.save(baseline, str(baseline_path))

    result = run_ann_experiment(
        {
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
            "outputs": {"runs_dir": str(tmp_path / "runs")},
        },
        _prepared_ann_features(),
    )

    artifacts = ArtifactLoader().load(
        ArtifactPaths(
            detector=detector_path,
            extractor_config=extractor_config_path,
            classifier=None,
            normalizer=normalizer_path,
            baseline=baseline_path,
            classifier_run_dir=Path(result["output_dir"]),
        ),
        config={"classifier": {"type": "ann"}},
    )

    assert isinstance(artifacts.classifier, torch.nn.Module)
    logits = artifacts.classifier(
        torch.tensor(_prepared_ann_features().val.features, dtype=torch.float32)
    )
    assert logits.shape == (3, 3)
    assert artifacts.classifier_run_summary is not None
    assert artifacts.classifier_run_summary["classifier_name"] == "ann"


def test_artifact_loader_restores_svm_from_run_dir(tmp_path: Path) -> None:
    """ArtifactLoader must restore an SVM classifier from a persisted run dir."""
    detector_path = tmp_path / "detector.pt"
    extractor_config_path = tmp_path / "extractor_config.json"
    normalizer_path = tmp_path / "normalizer.npz"
    baseline_path = tmp_path / "baseline.json"

    detector_path.write_text("detector", encoding="utf-8")
    extractor_config_path.write_text("{}", encoding="utf-8")
    normalizer_path.write_text("normalizer", encoding="utf-8")

    baseline = BaselineEstimator(["Platelets", "RBC", "WBC"]).fit_from_train_counts(
        train_counts=[{"Platelets": 1, "RBC": 8, "WBC": 1}],
        config={"classifier": {"type": "svm"}},
    )
    BaselineRepository.save(baseline, str(baseline_path))

    prepared = _prepared_svm_features()
    result = run_svm_experiment(
        {
            "svm": {
                "kernel": "linear",
                "C": 1.0,
                "gamma": "scale",
            },
            "outputs": {"runs_dir": str(tmp_path / "runs")},
        },
        prepared,
    )

    artifacts = ArtifactLoader().load(
        ArtifactPaths(
            detector=detector_path,
            extractor_config=extractor_config_path,
            classifier=None,
            normalizer=normalizer_path,
            baseline=baseline_path,
            classifier_run_dir=Path(result["output_dir"]),
        ),
        config={"classifier": {"type": "svm"}},
    )

    predictions = artifacts.classifier.predict(prepared.val.features)
    assert predictions.shape == (3,)
    assert artifacts.classifier_run_summary is not None
    assert artifacts.classifier_run_summary["classifier_name"] == "svm"
