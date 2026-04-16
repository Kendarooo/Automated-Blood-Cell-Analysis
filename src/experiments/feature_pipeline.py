"""Feature preparation pipeline for FR-11 sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from src.features.extractor import LAYER_DIMS
from src.features.normalize import FeatureNormalizer


class SplitDataLoaderProtocol(Protocol):
    """Protocol for loading one dataset split as tensors and labels."""

    def load_split(
        self,
        split_dir: Path,
        class_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return images tensor and encoded labels for one split."""


class ExtractorProtocol(Protocol):
    """Protocol for feature extractors compatible with the pipeline."""

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of produced feature vectors."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to a batch of feature vectors."""


@dataclass(frozen=True)
class FeaturePipelineConfig:
    """Typed configuration required to prepare train/validation features."""

    train_dir: Path
    val_dir: Path
    truncate_at: str
    projection_dim: int | None
    normalization_enabled: bool
    dimensionality_strategy: str
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSplit:
    """Features and encoded labels for one dataset split."""

    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata describing the prepared feature space."""

    class_names: tuple[str, ...]
    feature_dim: int
    truncate_at: str
    projection_dim: int | None
    normalization_enabled: bool
    dimensionality_strategy: str


@dataclass(frozen=True)
class PreparedFeatureSplits:
    """Prepared train/validation feature splits plus their metadata."""

    train: FeatureSplit
    val: FeatureSplit
    metadata: FeatureMetadata


class FeaturePipeline:
    """Prepare train/validation features with train-only normalization."""

    def __init__(
        self,
        config: FeaturePipelineConfig,
        split_loader: SplitDataLoaderProtocol,
        extractor: ExtractorProtocol,
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        self._config = config
        self._split_loader = split_loader
        self._extractor = extractor
        self._normalizer = normalizer or FeatureNormalizer()
        self._validate_config()

    def prepare_train_val_features(self) -> PreparedFeatureSplits:
        """Prepare train/validation features without leaking validation data."""
        train_images, train_labels = self._split_loader.load_split(
            self._config.train_dir,
            self._config.class_names,
        )
        val_images, val_labels = self._split_loader.load_split(
            self._config.val_dir,
            self._config.class_names,
        )

        train_features = self._extract_split_features(train_images)
        val_features = self._extract_split_features(val_images)

        if train_features.shape[1] != val_features.shape[1]:
            raise RuntimeError(
                "Train and validation features must share the same dimensionality."
            )

        if self._config.normalization_enabled:
            train_features = self._normalizer.fit_transform(train_features)
            val_features = self._normalizer.transform(val_features)

        metadata = FeatureMetadata(
            class_names=self._config.class_names,
            feature_dim=int(train_features.shape[1]),
            truncate_at=self._config.truncate_at,
            projection_dim=self._config.projection_dim,
            normalization_enabled=self._config.normalization_enabled,
            dimensionality_strategy=self._config.dimensionality_strategy,
        )

        return PreparedFeatureSplits(
            train=FeatureSplit(features=train_features, labels=train_labels),
            val=FeatureSplit(features=val_features, labels=val_labels),
            metadata=metadata,
        )

    def _extract_split_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract one feature vector per image."""
        features = self._extractor(images)
        return features.detach().cpu().numpy()

    def _validate_config(self) -> None:
        """Validate configuration before any heavy pipeline work starts."""
        if self._config.truncate_at not in LAYER_DIMS:
            raise ValueError(
                f"Unsupported truncate_at='{self._config.truncate_at}'. "
                f"Expected one of {list(LAYER_DIMS.keys())}."
            )

        if self._config.dimensionality_strategy not in {"none", "projection"}:
            raise ValueError(
                "Unsupported dimensionality strategy. "
                "Expected 'none' or 'projection'."
            )

        if self._config.dimensionality_strategy == "projection":
            if self._config.projection_dim is None:
                raise ValueError(
                    "projection_dim must be provided when "
                    "dimensionality_strategy='projection'."
                )
            if self._config.projection_dim <= 0:
                raise ValueError("projection_dim must be a positive integer.")
        elif self._config.projection_dim is not None:
            raise ValueError(
                "projection_dim must be None when dimensionality_strategy='none'."
            )

        if not self._config.class_names:
            raise ValueError("class_names must contain at least one class.")
