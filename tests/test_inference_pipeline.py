"""Integration-style tests for the FR-13 inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.detection.yolo_infer import BoundingBox, DetectionResult
from src.features.normalize import FeatureNormalizer
from src.inference.artifact_loader import ArtifactPaths, InferenceArtifacts
from src.inference.baseline import BaselineDistribution
from src.inference.pipeline import InferencePipeline


class StubDetector:
    """Detector stub that returns prebuilt boxes and crops."""

    def predict(self, image_path: Path) -> DetectionResult:
        crop_a = Image.new("RGB", (224, 224), color=(255, 0, 0))
        crop_b = Image.new("RGB", (224, 224), color=(0, 255, 0))
        return DetectionResult(
            image_path=image_path,
            boxes=[
                BoundingBox(0, 0, 10, 10, 0.9, 1, "RBC"),
                BoundingBox(10, 10, 20, 20, 0.8, 2, "WBC"),
            ],
            crops=[crop_a, crop_b],
        )


class StubArtifactLoader:
    """Artifact loader stub that injects ready-to-use restored artifacts."""

    def __init__(self, artifacts: InferenceArtifacts) -> None:
        self._artifacts = artifacts

    def load(self, paths: ArtifactPaths, config: dict) -> InferenceArtifacts:
        del paths, config
        return self._artifacts


class StubExtractor:
    """Extractor stub that returns deterministic two-dimensional features."""

    def eval(self) -> "StubExtractor":
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        del inputs
        return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)


class StubANN(torch.nn.Module):
    """ANN stub that predicts RBC then WBC."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        del features
        return torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        )


class StubSVM:
    """SVM stub that predicts RBC then WBC."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        del features
        return np.array([1, 2], dtype=np.int64)


def _cfg() -> dict:
    return {
        "augmentation": {
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "extractor": {
            "truncate_at": "layer3",
            "projection_dim": None,
        },
        "inference": {
            "alpha": 0.05,
            "min_cells_for_test": 1,
            "min_expected_count": 0.1,
            "test_type": "chi_square",
        },
        "detection": {
            "best_weights": "unused.pt",
            "imgsz": 320,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "crop_width": 224,
            "crop_height": 224,
        },
    }


def _artifacts(classifier: object) -> InferenceArtifacts:
    normalizer = FeatureNormalizer().fit(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    baseline = BaselineDistribution(
        class_order=["Platelets", "RBC", "WBC"],
        proportions={"Platelets": 0.1, "RBC": 0.8, "WBC": 0.1},
        config_fingerprint="unused",
        created_from_split="train",
    )
    return InferenceArtifacts(
        detector=Path("detector.pt"),
        extractor_config=Path("extractor.json"),
        classifier=classifier,
        normalizer=normalizer,
        baseline=baseline,
        classifier_run_summary={"classifier_name": "stub"},
    )


def test_inference_pipeline_runs_end_to_end_with_svm_classifier(monkeypatch, tmp_path: Path) -> None:
    """Pipeline should detect, classify, aggregate, and test one image end-to-end."""
    monkeypatch.setattr("src.inference.pipeline.ResNet18Extractor", lambda cfg: StubExtractor())

    pipeline = InferencePipeline(
        _cfg(),
        ArtifactPaths(
            detector=tmp_path / "detector.pt",
            extractor_config=tmp_path / "extractor.json",
            classifier=None,
            normalizer=tmp_path / "normalizer.npz",
            baseline=tmp_path / "baseline.json",
        ),
        artifact_loader=StubArtifactLoader(_artifacts(StubSVM())),
        detector=StubDetector(),
    )

    result = pipeline.predict(tmp_path / "image.jpg")

    assert result.num_detections == 2
    assert [prediction.classifier_label for prediction in result.predictions] == ["RBC", "WBC"]
    assert result.statistics.counts == {"Platelets": 0, "RBC": 1, "WBC": 1}
    assert result.statistical_test.test_executed


def test_inference_pipeline_supports_ann_classifier(monkeypatch, tmp_path: Path) -> None:
    """Pipeline should also accept restored ANN models for classification."""
    monkeypatch.setattr("src.inference.pipeline.ResNet18Extractor", lambda cfg: StubExtractor())

    pipeline = InferencePipeline(
        _cfg(),
        ArtifactPaths(
            detector=tmp_path / "detector.pt",
            extractor_config=tmp_path / "extractor.json",
            classifier=None,
            normalizer=tmp_path / "normalizer.npz",
            baseline=tmp_path / "baseline.json",
        ),
        artifact_loader=StubArtifactLoader(_artifacts(StubANN())),
        detector=StubDetector(),
    )

    result = pipeline.predict(tmp_path / "image.jpg")

    assert [prediction.classifier_label for prediction in result.predictions] == ["RBC", "WBC"]
    assert result.statistics.total_cells == 2
