"""Label record types, confusion-matrix builder, and I/O utilities for evaluation."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from src.evaluation.iou_matching import GroundTruthBox, PredictedBox

if TYPE_CHECKING:
    from src.detection.yolo_infer import DetectionResult
    from src.inference.pipeline import InferenceResult


@dataclass(frozen=True)
class MatchRecord:
    """One comparison record written into the evaluation CSV."""

    image_path: str
    case_type: str
    ground_truth_label: str
    yolo_label: str
    classifier_label: str
    confidence: float | None
    iou: float | None
    p_value: float | None
    alert: bool
    test_executed: bool


def build_confusion_matrix(
    records: Iterable[MatchRecord],
    *,
    actual_getter,
    predicted_getter,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Build a labeled confusion matrix from match records."""
    records_list = list(records)
    labels = sorted(
        {actual_getter(r) for r in records_list}
        | {predicted_getter(r) for r in records_list}
    )
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    for record in records_list:
        matrix[actual_getter(record)][predicted_getter(record)] += 1
    return labels, matrix


def write_confusion_csv(
    path: Path,
    confusion: tuple[list[str], dict[str, dict[str, int]]],
) -> None:
    """Persist one confusion matrix as a CSV file."""
    labels, matrix = confusion
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for actual in labels:
            writer.writerow([actual, *[matrix[actual][pred] for pred in labels]])


def write_per_image_csv(path: Path, records: list[MatchRecord]) -> None:
    """Persist detailed per-image/per-record comparison rows as CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path", "case_type", "ground_truth_label",
                "yolo_label", "classifier_label", "confidence",
                "iou", "p_value", "alert", "test_executed",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_path": record.image_path,
                "case_type": record.case_type,
                "ground_truth_label": record.ground_truth_label,
                "yolo_label": record.yolo_label,
                "classifier_label": record.classifier_label,
                "confidence": record.confidence,
                "iou": record.iou,
                "p_value": record.p_value,
                "alert": record.alert,
                "test_executed": record.test_executed,
            })


def find_label_path(labels_dir: Path, stem: str) -> Path:
    """Resolve the label file path for a given image stem."""
    label_path = labels_dir / f"{stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Could not find label file for image stem '{stem}'."
        )
    return label_path


def load_ground_truth_boxes(
    *,
    label_path: Path,
    image_path: Path,
    class_names: list[str],
) -> list[GroundTruthBox]:
    """Load one BCCD YOLO label file and convert boxes to absolute pixels."""
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.width, image.height

    boxes: list[GroundTruthBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        class_id, x_center, y_center, box_width, box_height = stripped.split()
        x1, y1, x2, y2 = _to_pixel_box(
            x_center=float(x_center),
            y_center=float(y_center),
            box_width=float(box_width),
            box_height=float(box_height),
            image_width=width,
            image_height=height,
        )
        boxes.append(
            GroundTruthBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_names[int(class_id)])
        )
    return boxes


def build_predicted_boxes(
    detection_result: DetectionResult,
    inference_result: InferenceResult,
) -> list[PredictedBox]:
    """Merge detector boxes with pipeline predictions into typed structures."""
    return [
        PredictedBox(
            x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
            yolo_label=prediction.yolo_label,
            classifier_label=prediction.classifier_label,
            confidence=prediction.confidence,
        )
        for box, prediction in zip(
            detection_result.boxes, inference_result.predictions, strict=True
        )
    ]


def summarize_image(image_path: Path, records: list[MatchRecord]) -> dict[str, object]:
    """Build a per-image summary dict for the JSON evaluation report."""
    return {
        "image_path": str(image_path),
        "num_records": len(records),
        "num_matched": sum(r.case_type == "matched" for r in records),
        "num_false_negatives": sum(r.case_type == "false_negative" for r in records),
        "num_false_positives": sum(r.case_type == "false_positive" for r in records),
        "p_value": records[0].p_value if records else None,
        "alert": records[0].alert if records else False,
        "test_executed": records[0].test_executed if records else False,
    }


def _to_pixel_box(
    *,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO coordinates to absolute pixel coordinates."""
    x1 = int(round((x_center - box_width / 2.0) * image_width))
    y1 = int(round((y_center - box_height / 2.0) * image_height))
    x2 = int(round((x_center + box_width / 2.0) * image_width))
    y2 = int(round((y_center + box_height / 2.0) * image_height))
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2
