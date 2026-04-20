"""Artifact loading utilities for FR-10 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.inference.baseline import BaselineDistribution, BaselineRepository
from src.inference.run_loader import load_run
from src.features.normalize import FeatureNormalizer


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem locations of the persisted inference artifacts."""

    detector: Path
    extractor_config: Path
    classifier: Path | None
    normalizer: Path
    baseline: Path
    classifier_run_dir: Path | None = None


@dataclass(frozen=True)
class InferenceArtifacts:
    """
    Loaded artifacts required by the inference pipeline.

    Detector/extractor_config/normalizer remain as validated paths.
    The classifier may be either a validated artifact path or a restored model
    loaded from a persisted experiment run.
    """

    detector: Path
    extractor_config: Path
    classifier: object
    normalizer: FeatureNormalizer
    baseline: BaselineDistribution
    classifier_run_summary: dict[str, Any] | None = None


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
        self._require_existing_path(paths.normalizer)
        self._require_existing_path(paths.baseline)

        baseline = BaselineRepository.load(
            str(paths.baseline),
            expected_config=config,
        )
        normalizer = FeatureNormalizer.load(str(paths.normalizer))

        classifier: object
        classifier_run_summary: dict[str, Any] | None = None
        if paths.classifier_run_dir is not None:
            loaded_run = load_run(paths.classifier_run_dir)
            classifier = loaded_run.model
            classifier_run_summary = loaded_run.run_summary
        else:
            if paths.classifier is None:
                raise ValueError(
                    "classifier must be provided when classifier_run_dir is not set."
                )
            self._require_existing_path(paths.classifier)
            self._validate_classifier_type(
                classifier_path=paths.classifier,
                config=config,
            )
            classifier = paths.classifier

        return InferenceArtifacts(
            detector=paths.detector,
            extractor_config=paths.extractor_config,
            classifier=classifier,
            normalizer=normalizer,
            baseline=baseline,
            classifier_run_summary=classifier_run_summary,
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
