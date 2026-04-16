"""Tests for the FR-11 feature preparation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
)


class RecordingNormalizer:
    """Test double that records whether fit and transform were called."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.transform_calls = 0
        self.fit_inputs: list[np.ndarray] = []
        self.transform_inputs: list[np.ndarray] = []

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit_calls += 1
        self.fit_inputs.append(features.copy())
        return features + 10.0

    def transform(self, features: np.ndarray) -> np.ndarray:
        self.transform_calls += 1
        self.transform_inputs.append(features.copy())
        return features + 20.0


class StubSplitLoader:
    """Loader double that returns deterministic train/validation tensors."""

    def __init__(self) -> None:
        self.called_split_dirs: list[Path] = []

    def load_split(
        self,
        split_dir: Path,
        class_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, np.ndarray]:
        del class_names
        self.called_split_dirs.append(split_dir)

        if split_dir.name == "train":
            return torch.ones((3, 3, 224, 224)), np.array([0, 1, 2], dtype=np.int64)
        if split_dir.name == "val":
            return torch.full((2, 3, 224, 224), 2.0), np.array([2, 1], dtype=np.int64)

        raise AssertionError(f"Unexpected split requested: {split_dir}")


class StubExtractor:
    """Extractor double faithful to the real FR-5 dimensionality contract."""

    def __init__(self, truncate_at: str, projection_dim: int | None = None) -> None:
        base_dims = {
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }
        self._base_dim = base_dims[truncate_at]
        self.output_dim = projection_dim or self._base_dim

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        row = torch.arange(self.output_dim, dtype=torch.float32)
        return row.repeat(batch_size, 1)


def _config(
    *,
    truncate_at: str = "layer3",
    projection_dim: int | None = None,
    dimensionality_strategy: str = "none",
    normalization_enabled: bool = True,
) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        train_dir=Path("train"),
        val_dir=Path("val"),
        truncate_at=truncate_at,
        projection_dim=projection_dim,
        normalization_enabled=normalization_enabled,
        dimensionality_strategy=dimensionality_strategy,
        class_names=("WBC", "RBC", "Platelets"),
    )


def test_feature_pipeline_rejects_invalid_truncation() -> None:
    """Invalid truncation points must fail before any feature extraction."""
    with pytest.raises(ValueError, match="Unsupported truncate_at"):
        FeaturePipeline(
            config=_config(truncate_at="layer99"),
            split_loader=StubSplitLoader(),
            extractor=StubExtractor("layer2"),
        )


def test_feature_pipeline_rejects_projection_without_dimension() -> None:
    """Projection strategy must declare a positive bottleneck dimension."""
    with pytest.raises(ValueError, match="projection_dim must be provided"):
        FeaturePipeline(
            config=_config(
                truncate_at="layer3",
                projection_dim=None,
                dimensionality_strategy="projection",
            ),
            split_loader=StubSplitLoader(),
            extractor=StubExtractor("layer3"),
        )


@pytest.mark.parametrize(
    ("truncate_at", "expected_dim"),
    [
        ("layer2", 128),
        ("layer3", 256),
        ("layer4", 512),
    ],
)
def test_feature_pipeline_respects_fr5_feature_dimensions(
    truncate_at: str,
    expected_dim: int,
) -> None:
    """Feature dimensions must match the real extractor contract for FR-5."""
    pipeline = FeaturePipeline(
        config=_config(truncate_at=truncate_at),
        split_loader=StubSplitLoader(),
        extractor=StubExtractor(truncate_at),
    )

    prepared = pipeline.prepare_train_val_features()

    assert prepared.train.features.shape == (3, expected_dim)
    assert prepared.val.features.shape == (2, expected_dim)
    assert prepared.metadata.feature_dim == expected_dim


def test_feature_pipeline_fits_normalization_only_on_train() -> None:
    """Validation data must only be transformed, never used in fit."""
    normalizer = RecordingNormalizer()
    pipeline = FeaturePipeline(
        config=_config(truncate_at="layer3", normalization_enabled=True),
        split_loader=StubSplitLoader(),
        extractor=StubExtractor("layer3"),
        normalizer=normalizer,
    )

    prepared = pipeline.prepare_train_val_features()

    assert normalizer.fit_calls == 1
    assert normalizer.transform_calls == 1
    assert normalizer.fit_inputs[0].shape == (3, 256)
    assert normalizer.transform_inputs[0].shape == (2, 256)
    assert not np.array_equal(
        normalizer.fit_inputs[0],
        normalizer.transform_inputs[0],
    ), "fit y transform recibieron los mismos datos"
    assert np.all(prepared.train.features >= 10.0)
    assert np.all(prepared.val.features >= 20.0)


def test_feature_pipeline_metadata_reports_projection_consistently() -> None:
    """Metadata must expose the effective projected dimensionality."""
    pipeline = FeaturePipeline(
        config=_config(
            truncate_at="layer4",
            projection_dim=64,
            dimensionality_strategy="projection",
        ),
        split_loader=StubSplitLoader(),
        extractor=StubExtractor("layer4", projection_dim=64),
    )

    prepared = pipeline.prepare_train_val_features()

    assert prepared.metadata.truncate_at == "layer4"
    assert prepared.metadata.projection_dim == 64
    assert prepared.metadata.feature_dim == 64
    assert prepared.metadata.class_names == ("WBC", "RBC", "Platelets")


def test_feature_pipeline_only_requests_train_and_val_splits() -> None:
    """The feature pipeline contract must not touch the test split."""
    split_loader = StubSplitLoader()
    pipeline = FeaturePipeline(
        config=_config(truncate_at="layer2"),
        split_loader=split_loader,
        extractor=StubExtractor("layer2"),
    )

    pipeline.prepare_train_val_features()

    requested = [path.name for path in split_loader.called_split_dirs]
    assert requested == ["train", "val"]
