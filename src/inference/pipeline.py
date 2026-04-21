"""End-to-end inference pipeline for FR-13 using persisted artifacts."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from src.data.transforms import ValTransforms
from src.detection.yolo_infer import DetectionResult, YOLOInferencer
from src.features.extractor import ResNet18Extractor
from src.inference.artifact_loader import ArtifactLoader, ArtifactPaths
from src.inference.cell_statistics import (
    CellCountAggregator,
    CellProportionCalculator,
    CellStatistics,
)
from src.inference.statistical_test import (
    StatisticalAnomalyDetector,
    StatisticalTestConfig,
    StatisticalTestResult,
)


class DetectorProtocol(Protocol):
    """Protocol for object detectors used by the inference pipeline."""

    def predict(self, image_path: Path) -> DetectionResult:
        """Return boxes and crops for one microscopy image."""


@dataclass(frozen=True)
class InferencePrediction:
    """Per-cell prediction after detection and classification."""

    yolo_label: str
    classifier_label: str
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    """Structured output of one end-to-end inference pass."""

    image_path: Path
    num_detections: int
    predictions: list[InferencePrediction]
    statistics: CellStatistics
    statistical_test: StatisticalTestResult
    classifier_name: str | None = None


class InferencePipeline:
    """Run detection, feature extraction, classification, and statistical testing."""

    def __init__(
        self,
        cfg: dict[str, Any],
        artifact_paths: ArtifactPaths,
        *,
        artifact_loader: ArtifactLoader | None = None,
        detector: DetectorProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        self._artifacts = (artifact_loader or ArtifactLoader()).load(artifact_paths, cfg)
        self._detector = detector or YOLOInferencer(cfg)
        self._extractor = ResNet18Extractor(cfg).eval()
        self._transform = ValTransforms(cfg)
        self._count_aggregator = CellCountAggregator(
            list(self._artifacts.baseline.class_order)
        )
        self._proportion_calculator = CellProportionCalculator(
            list(self._artifacts.baseline.class_order)
        )
        inference_cfg = cfg.get("inference", {})
        self._detector_stats = StatisticalAnomalyDetector(
            StatisticalTestConfig(
                alpha=float(inference_cfg.get("alpha", 0.05)),
                min_cells_for_test=int(inference_cfg.get("min_cells_for_test", 5)),
                min_expected_count=float(inference_cfg.get("min_expected_count", 1.0)),
                test_type=str(inference_cfg.get("test_type", "chi_square")),
            )
        )

    def predict(self, image_path: str | Path) -> InferenceResult:
        """Run the full pipeline on one microscopy image."""
        resolved_image_path = Path(image_path)
        detection = self._detector.predict(resolved_image_path)
        classifier_labels = self._classify_crops(detection)

        predictions = [
            InferencePrediction(
                yolo_label=box.class_name,
                classifier_label=label,
                confidence=box.confidence,
            )
            for box, label in zip(detection.boxes, classifier_labels, strict=True)
        ]
        counts = self._count_aggregator.aggregate(classifier_labels)
        statistics = self._proportion_calculator.compute(counts)
        test_result = self._detector_stats.evaluate(
            baseline=self._artifacts.baseline,
            statistics=statistics,
        )

        classifier_name = None
        if self._artifacts.classifier_run_summary is not None:
            classifier_name = self._artifacts.classifier_run_summary.get("classifier_name")

        return InferenceResult(
            image_path=resolved_image_path,
            num_detections=len(detection.boxes),
            predictions=predictions,
            statistics=statistics,
            statistical_test=test_result,
            classifier_name=classifier_name,
        )

    def _classify_crops(self, detection: DetectionResult) -> list[str]:
        """Classify all crops using the restored classifier artifacts."""
        if not detection.crops:
            return []

        image_batch = torch.stack([self._transform(crop) for crop in detection.crops])
        with torch.no_grad():
            features = self._extractor(image_batch).detach().cpu().numpy()
        normalized_features = self._artifacts.normalizer.transform(features)
        class_ids = self._predict_class_ids(normalized_features)
        class_order = list(self._artifacts.baseline.class_order)
        return [class_order[class_id] for class_id in class_ids]

    def _predict_class_ids(self, features: np.ndarray) -> np.ndarray:
        """Predict encoded class ids from normalized feature vectors."""
        classifier = self._artifacts.classifier
        if hasattr(classifier, "predict") and not isinstance(classifier, torch.nn.Module):
            return np.asarray(classifier.predict(features), dtype=np.int64)

        if isinstance(classifier, torch.nn.Module):
            with torch.no_grad():
                logits = classifier(torch.tensor(features, dtype=torch.float32))
            return logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

        raise TypeError("Unsupported classifier artifact for inference.")
