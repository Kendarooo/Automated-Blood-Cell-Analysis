"""Compare ground truth, YOLO labels, and classifier labels on one BCCD split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.inference.artifact_loader import ArtifactPaths
from src.inference.pipeline import InferencePipeline, InferenceResult
from src.utils.config import load_config


BACKGROUND_LABEL = "__missing__"


@dataclass(frozen=True)
class GroundTruthBox:
    """Ground-truth box and label in absolute pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str


@dataclass(frozen=True)
class PredictedBox:
    """Predicted box enriched with both YOLO and classifier labels."""

    x1: int
    y1: int
    x2: int
    y2: int
    yolo_label: str
    classifier_label: str
    confidence: float


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


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for label comparison over one full BCCD split."""
    parser = argparse.ArgumentParser(
        description="Compare ground truth, YOLO, and classifier labels over BCCD test.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project configuration file.",
    )
    parser.add_argument(
        "--classifier-run-dir",
        required=True,
        help="Persisted ANN/SVM run directory produced by FR-11.",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/inference/baseline.json",
        help="Baseline JSON generated from the training split.",
    )
    parser.add_argument(
        "--test-dir",
        default="data/bccd/test",
        help="BCCD split directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/compare_labels",
        help="Directory where CSV/JSON artifacts will be written.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Optional YOLO confidence threshold override for this evaluation run.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run label comparison over one full BCCD split."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.conf_threshold is not None:
        cfg["detection"]["conf_threshold"] = args.conf_threshold
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_run_dir = Path(args.classifier_run_dir)
    pipeline = InferencePipeline(
        cfg,
        ArtifactPaths(
            detector=Path(cfg["detection"]["best_weights"]),
            extractor_config=classifier_run_dir / "effective_config.json",
            classifier=None,
            normalizer=classifier_run_dir / "normalizer.npz",
            baseline=Path(args.baseline),
            classifier_run_dir=classifier_run_dir,
        ),
    )

    test_dir = Path(args.test_dir)
    image_paths = sorted((test_dir / "images").iterdir())
    records: list[MatchRecord] = []
    per_image_summary: list[dict[str, object]] = []

    for image_path in image_paths:
        label_path = _find_label_path(test_dir / "labels", image_path.stem)
        detection = pipeline._detector.predict(image_path)  # noqa: SLF001
        inference_result = pipeline.predict(image_path)
        gt_boxes = _load_ground_truth_boxes(
            label_path=label_path,
            image_path=image_path,
            class_names=cfg["dataset"]["class_names"],
        )
        predicted_boxes = _build_predicted_boxes(detection, inference_result)
        image_records = compare_image(
            image_path=image_path,
            ground_truth_boxes=gt_boxes,
            predicted_boxes=predicted_boxes,
            inference_result=inference_result,
            iou_threshold=args.iou_threshold,
        )
        records.extend(image_records)
        per_image_summary.append(_summarize_image(image_path, image_records))

    per_image_csv = output_dir / "per_image.csv"
    _write_per_image_csv(per_image_csv, records)

    confusion_yolo_vs_gt = build_confusion_matrix(
        records,
        actual_getter=lambda record: record.ground_truth_label,
        predicted_getter=lambda record: record.yolo_label,
    )
    confusion_classifier_vs_gt = build_confusion_matrix(
        records,
        actual_getter=lambda record: record.ground_truth_label,
        predicted_getter=lambda record: record.classifier_label,
    )
    confusion_yolo_vs_classifier = build_confusion_matrix(
        [
            record
            for record in records
            if record.yolo_label != BACKGROUND_LABEL
            and record.classifier_label != BACKGROUND_LABEL
        ],
        actual_getter=lambda record: record.yolo_label,
        predicted_getter=lambda record: record.classifier_label,
    )

    yolo_vs_gt_path = output_dir / "confusion_yolo_vs_gt.csv"
    classifier_vs_gt_path = output_dir / "confusion_classifier_vs_gt.csv"
    yolo_vs_classifier_path = output_dir / "confusion_yolo_vs_classifier.csv"
    _write_confusion_csv(yolo_vs_gt_path, confusion_yolo_vs_gt)
    _write_confusion_csv(classifier_vs_gt_path, confusion_classifier_vs_gt)
    _write_confusion_csv(yolo_vs_classifier_path, confusion_yolo_vs_classifier)

    summary = {
        "split_dir": str(test_dir),
        "conf_threshold": cfg["detection"]["conf_threshold"],
        "num_test_images": len(image_paths),
        "num_records": len(records),
        "output_dir": str(output_dir),
        "per_image_csv": str(per_image_csv),
        "confusion_yolo_vs_gt": str(yolo_vs_gt_path),
        "confusion_classifier_vs_gt": str(classifier_vs_gt_path),
        "confusion_yolo_vs_classifier": str(yolo_vs_classifier_path),
        "image_summaries": per_image_summary,
        "record_counts": dict(Counter(record.case_type for record in records)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def compare_image(
    *,
    image_path: Path,
    ground_truth_boxes: list[GroundTruthBox],
    predicted_boxes: list[PredictedBox],
    inference_result: InferenceResult,
    iou_threshold: float,
) -> list[MatchRecord]:
    """Compare one image worth of ground truth boxes and predicted boxes."""
    all_records: list[MatchRecord] = []
    class_names = sorted({box.label for box in ground_truth_boxes} | {box.yolo_label for box in predicted_boxes})

    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()

    for class_name in class_names:
        gt_subset = [
            (index, box)
            for index, box in enumerate(ground_truth_boxes)
            if box.label == class_name
        ]
        pred_subset = [
            (index, box)
            for index, box in enumerate(predicted_boxes)
            if box.yolo_label == class_name
        ]
        matches = greedy_iou_match(gt_subset, pred_subset, threshold=iou_threshold)
        for gt_index, pred_index, iou_value in matches:
            matched_gt_indices.add(gt_index)
            matched_pred_indices.add(pred_index)
            gt_box = ground_truth_boxes[gt_index]
            pred_box = predicted_boxes[pred_index]
            all_records.append(
                MatchRecord(
                    image_path=str(image_path),
                    case_type="matched",
                    ground_truth_label=gt_box.label,
                    yolo_label=pred_box.yolo_label,
                    classifier_label=pred_box.classifier_label,
                    confidence=pred_box.confidence,
                    iou=iou_value,
                    p_value=inference_result.statistical_test.p_value,
                    alert=inference_result.statistical_test.alert,
                    test_executed=inference_result.statistical_test.test_executed,
                )
            )

    for index, gt_box in enumerate(ground_truth_boxes):
        if index in matched_gt_indices:
            continue
        all_records.append(
            MatchRecord(
                image_path=str(image_path),
                case_type="false_negative",
                ground_truth_label=gt_box.label,
                yolo_label=BACKGROUND_LABEL,
                classifier_label=BACKGROUND_LABEL,
                confidence=None,
                iou=None,
                p_value=inference_result.statistical_test.p_value,
                alert=inference_result.statistical_test.alert,
                test_executed=inference_result.statistical_test.test_executed,
            )
        )

    for index, pred_box in enumerate(predicted_boxes):
        if index in matched_pred_indices:
            continue
        all_records.append(
            MatchRecord(
                image_path=str(image_path),
                case_type="false_positive",
                ground_truth_label=BACKGROUND_LABEL,
                yolo_label=pred_box.yolo_label,
                classifier_label=pred_box.classifier_label,
                confidence=pred_box.confidence,
                iou=None,
                p_value=inference_result.statistical_test.p_value,
                alert=inference_result.statistical_test.alert,
                test_executed=inference_result.statistical_test.test_executed,
            )
        )

    return all_records


def greedy_iou_match(
    ground_truth_boxes: list[tuple[int, GroundTruthBox]],
    predicted_boxes: list[tuple[int, PredictedBox]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one IoU matching restricted to the same class."""
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_box in ground_truth_boxes:
        for pred_index, pred_box in predicted_boxes:
            iou_value = compute_iou(gt_box, pred_box)
            if iou_value >= threshold:
                candidates.append((iou_value, gt_index, pred_index))

    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    for iou_value, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou_value))

    return matches


def compute_iou(box_a: GroundTruthBox, box_b: PredictedBox) -> float:
    """Compute intersection-over-union between two absolute boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def build_confusion_matrix(
    records: Iterable[MatchRecord],
    *,
    actual_getter,
    predicted_getter,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Build a labeled confusion matrix from match records."""
    records_list = list(records)
    labels = sorted(
        {
            actual_getter(record)
            for record in records_list
        }
        | {
            predicted_getter(record)
            for record in records_list
        }
    )
    matrix = {
        actual_label: {predicted_label: 0 for predicted_label in labels}
        for actual_label in labels
    }
    for record in records_list:
        matrix[actual_getter(record)][predicted_getter(record)] += 1
    return labels, matrix


def _write_confusion_csv(
    path: Path,
    confusion: tuple[list[str], dict[str, dict[str, int]]],
) -> None:
    """Persist one confusion matrix as CSV."""
    labels, matrix = confusion
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for actual_label in labels:
            writer.writerow([actual_label, *[matrix[actual_label][predicted_label] for predicted_label in labels]])


def _write_per_image_csv(path: Path, records: list[MatchRecord]) -> None:
    """Persist detailed per-image/per-record comparison rows."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "case_type",
                "ground_truth_label",
                "yolo_label",
                "classifier_label",
                "confidence",
                "iou",
                "p_value",
                "alert",
                "test_executed",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
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
                }
            )


def _load_ground_truth_boxes(
    *,
    label_path: Path,
    image_path: Path,
    class_names: list[str],
) -> list[GroundTruthBox]:
    """Load one BCCD YOLO label file and convert boxes to absolute pixels."""
    from PIL import Image

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
            GroundTruthBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=class_names[int(class_id)],
            )
        )
    return boxes


def _build_predicted_boxes(detection_result, inference_result: InferenceResult) -> list[PredictedBox]:
    """Merge detector boxes with pipeline predictions in one typed structure."""
    return [
        PredictedBox(
            x1=box.x1,
            y1=box.y1,
            x2=box.x2,
            y2=box.y2,
            yolo_label=prediction.yolo_label,
            classifier_label=prediction.classifier_label,
            confidence=prediction.confidence,
        )
        for box, prediction in zip(detection_result.boxes, inference_result.predictions, strict=True)
    ]


def _find_label_path(labels_dir: Path, stem: str) -> Path:
    """Resolve the test label path corresponding to one image stem."""
    label_path = labels_dir / f"{stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"Could not find label file for image stem '{stem}'.")
    return label_path


def _to_pixel_box(
    *,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO boxes to absolute pixel coordinates."""
    x1 = int(round((x_center - box_width / 2.0) * image_width))
    y1 = int(round((y_center - box_height / 2.0) * image_height))
    x2 = int(round((x_center + box_width / 2.0) * image_width))
    y2 = int(round((y_center + box_height / 2.0) * image_height))
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def _summarize_image(image_path: Path, records: list[MatchRecord]) -> dict[str, object]:
    """Build one per-image summary for the JSON report."""
    return {
        "image_path": str(image_path),
        "num_records": len(records),
        "num_matched": sum(record.case_type == "matched" for record in records),
        "num_false_negatives": sum(
            record.case_type == "false_negative" for record in records
        ),
        "num_false_positives": sum(
            record.case_type == "false_positive" for record in records
        ),
        "p_value": records[0].p_value if records else None,
        "alert": records[0].alert if records else False,
        "test_executed": records[0].test_executed if records else False,
    }


if __name__ == "__main__":
    main()
