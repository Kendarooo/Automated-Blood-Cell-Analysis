"""Artifact loading utilities for FR-10 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.inference.baseline import BaselineDistribution, BaselineRepository


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem locations of the persisted inference artifacts."""

    detector: Path
    extractor_config: Path
    classifier: Path
    normalizer: Path
    baseline: Path


@dataclass(frozen=True)
class InferenceArtifacts:
    """
    Loaded artifacts required by the inference pipeline.

    At this stage, detector/classifier/normalizer are represented by their
    validated paths until their dedicated loaders are integrated.
    """

    detector: Path
    extractor_config: Path
    classifier: Path
    normalizer: Path
    baseline: BaselineDistribution


class ArtifactLoader:
    """Load inference artifacts from disk and validate their availability."""

    def load(self, paths: ArtifactPaths, config: dict) -> InferenceArtifacts:
        """
        Load and validate inference artifacts.

        Args:
            paths: Locations of persisted artifacts.
            config: Current project configuration for baseline validation.

        Returns:
            InferenceArtifacts ready for the next pipeline stage.
        """
        self._require_existing_path(paths.detector)
        self._require_existing_path(paths.extractor_config)
        self._require_existing_path(paths.classifier)
        self._require_existing_path(paths.normalizer)
        self._require_existing_path(paths.baseline)
        self._validate_classifier_type(
            classifier_path=paths.classifier,
            config=config,
        )

        baseline = BaselineRepository.load(
            str(paths.baseline),
            expected_config=config,
        )

        return InferenceArtifacts(
            detector=paths.detector,
            extractor_config=paths.extractor_config,
            classifier=paths.classifier,
            normalizer=paths.normalizer,
            baseline=baseline,
        )

    @staticmethod
    def _require_existing_path(path: Path) -> None:
        """Raise a clear error if an artifact path does not exist."""
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

    @staticmethod
    def _validate_classifier_type(classifier_path: Path, config: dict) -> None:
        """
        Validate that the classifier artifact matches the configured classifier type.

        ANN artifacts are expected as `.pt` files.
        SVM artifacts are expected as `.joblib` or `.pkl` files.
        """
        classifier_cfg = config.get("classifier", {})
        classifier_type = classifier_cfg.get("type")

        if classifier_type is None:
            return

        suffix = classifier_path.suffix.lower()
        if classifier_type == "ann" and suffix != ".pt":
            raise ValueError(
                f"Unexpected classifier type for ANN: {classifier_path}. "
                "Expected a .pt artifact."
            )
        if classifier_type == "svm" and suffix not in {".joblib", ".pkl"}:
            raise ValueError(
                f"Unexpected classifier type for SVM: {classifier_path}. "
                "Expected a .joblib or .pkl artifact."
            )
