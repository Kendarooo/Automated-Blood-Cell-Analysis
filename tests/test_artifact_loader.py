"""Unit tests for artifact loading in FR-10 inference."""

from pathlib import Path

import pytest

from src.inference.baseline import BaselineEstimator, BaselineRepository
from src.inference.artifact_loader import ArtifactLoader, ArtifactPaths


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

    baseline = BaselineEstimator(["WBC", "RBC", "Platelets"]).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
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

    baseline = BaselineEstimator(["WBC", "RBC", "Platelets"]).fit_from_train_counts(
        train_counts=[{"WBC": 1, "RBC": 8, "Platelets": 1}],
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
