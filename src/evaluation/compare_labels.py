"""Compare ground truth, YOLO labels, and classifier labels on one BCCD split."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluation.iou_matching import (  # noqa: F401 (re-exported for callers)
    GroundTruthBox,
    PredictedBox,
    greedy_iou_match,
)
from src.evaluation.label_io import (  # noqa: F401 (re-exported for callers)
    MatchRecord,
    build_confusion_matrix,
    build_predicted_boxes,
    find_label_path,
    load_ground_truth_boxes,
    summarize_image,
    write_confusion_csv,
    write_per_image_csv,
)
from src.inference.artifact_loader import ArtifactPaths
from src.inference.pipeline import InferencePipeline, InferenceResult
from src.utils.config import load_config


BACKGROUND_LABEL = "__missing__"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for label comparison over one full BCCD split."""
    parser = argparse.ArgumentParser(
        description="Compare ground truth, YOLO, and classifier labels over BCCD test.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--classifier-run-dir", required=True)
    parser.add_argument("--baseline", default="outputs/inference/baseline.json")
    parser.add_argument("--test-dir", default="data/bccd/test")
    parser.add_argument("--output-dir", default="outputs/evaluation/compare_labels")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=None)
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
        label_path = find_label_path(test_dir / "labels", image_path.stem)
        detection = pipeline._detector.predict(image_path)  # noqa: SLF001
        inference_result = pipeline.predict(image_path)
        gt_boxes = load_ground_truth_boxes(
            label_path=label_path,
            image_path=image_path,
            class_names=cfg["dataset"]["class_names"],
        )
        predicted_boxes = build_predicted_boxes(detection, inference_result)
        image_records = compare_image(
            image_path=image_path,
            ground_truth_boxes=gt_boxes,
            predicted_boxes=predicted_boxes,
            inference_result=inference_result,
            iou_threshold=args.iou_threshold,
        )
        records.extend(image_records)
        per_image_summary.append(summarize_image(image_path, image_records))

    per_image_csv = output_dir / "per_image.csv"
    write_per_image_csv(per_image_csv, records)

    confusion_yolo_vs_gt = build_confusion_matrix(
        records,
        actual_getter=lambda r: r.ground_truth_label,
        predicted_getter=lambda r: r.yolo_label,
    )
    confusion_classifier_vs_gt = build_confusion_matrix(
        records,
        actual_getter=lambda r: r.ground_truth_label,
        predicted_getter=lambda r: r.classifier_label,
    )
    confusion_yolo_vs_classifier = build_confusion_matrix(
        [
            r for r in records
            if r.yolo_label != BACKGROUND_LABEL and r.classifier_label != BACKGROUND_LABEL
        ],
        actual_getter=lambda r: r.yolo_label,
        predicted_getter=lambda r: r.classifier_label,
    )

    yolo_vs_gt_path = output_dir / "confusion_yolo_vs_gt.csv"
    classifier_vs_gt_path = output_dir / "confusion_classifier_vs_gt.csv"
    yolo_vs_classifier_path = output_dir / "confusion_yolo_vs_classifier.csv"
    write_confusion_csv(yolo_vs_gt_path, confusion_yolo_vs_gt)
    write_confusion_csv(classifier_vs_gt_path, confusion_classifier_vs_gt)
    write_confusion_csv(yolo_vs_classifier_path, confusion_yolo_vs_classifier)

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
        "record_counts": dict(Counter(r.case_type for r in records)),
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
    class_names = sorted(
        {box.label for box in ground_truth_boxes}
        | {box.yolo_label for box in predicted_boxes}
    )

    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()

    for class_name in class_names:
        gt_subset = [(i, b) for i, b in enumerate(ground_truth_boxes) if b.label == class_name]
        pred_subset = [
            (i, b) for i, b in enumerate(predicted_boxes) if b.yolo_label == class_name
        ]
        matches = greedy_iou_match(gt_subset, pred_subset, threshold=iou_threshold)
        for gt_index, pred_index, iou_value in matches:
            matched_gt_indices.add(gt_index)
            matched_pred_indices.add(pred_index)
            gt_box = ground_truth_boxes[gt_index]
            pred_box = predicted_boxes[pred_index]
            all_records.append(MatchRecord(
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
            ))

    for index, gt_box in enumerate(ground_truth_boxes):
        if index in matched_gt_indices:
            continue
        all_records.append(MatchRecord(
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
        ))

    for index, pred_box in enumerate(predicted_boxes):
        if index in matched_pred_indices:
            continue
        all_records.append(MatchRecord(
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
        ))

    return all_records


if __name__ == "__main__":
    main()
