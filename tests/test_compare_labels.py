"""Tests for FR-14-style label comparison utilities."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.compare_labels import (
    BACKGROUND_LABEL,
    GroundTruthBox,
    MatchRecord,
    PredictedBox,
    build_confusion_matrix,
    compare_image,
    greedy_iou_match,
)
from src.inference.pipeline import InferenceResult
from src.inference.cell_statistics import CellStatistics
from src.inference.statistical_test import StatisticalTestResult


def _inference_result() -> InferenceResult:
    return InferenceResult(
        image_path=Path("image.jpg"),
        num_detections=2,
        predictions=[],
        statistics=CellStatistics(
            counts={"Platelets": 0, "RBC": 1, "WBC": 1},
            proportions={"Platelets": 0.0, "RBC": 0.5, "WBC": 0.5},
            total_cells=2,
        ),
        statistical_test=StatisticalTestResult(
            p_value=0.2,
            alert=False,
            test_executed=True,
            reason=None,
        ),
        classifier_name="ann",
    )


def test_greedy_iou_match_only_keeps_same_class_candidates() -> None:
    """Matching must be greedy, one-to-one, and class-restricted by caller."""
    gt_boxes = [
        (0, GroundTruthBox(0, 0, 10, 10, "RBC")),
        (1, GroundTruthBox(20, 20, 30, 30, "RBC")),
    ]
    pred_boxes = [
        (0, PredictedBox(0, 0, 10, 10, "RBC", "RBC", 0.9)),
        (1, PredictedBox(20, 20, 30, 30, "RBC", "RBC", 0.8)),
    ]

    matches = greedy_iou_match(gt_boxes, pred_boxes, threshold=0.5)

    assert matches == [(1, 1, 1.0), (0, 0, 1.0)] or matches == [(0, 0, 1.0), (1, 1, 1.0)]


def test_compare_image_records_matched_false_negatives_and_false_positives() -> None:
    """Comparison CSV rows must explicitly represent matched, FN, and FP cases."""
    ground_truth = [
        GroundTruthBox(0, 0, 10, 10, "RBC"),
        GroundTruthBox(20, 20, 30, 30, "Platelets"),
    ]
    predictions = [
        PredictedBox(0, 0, 10, 10, "RBC", "RBC", 0.9),
        PredictedBox(40, 40, 50, 50, "WBC", "WBC", 0.8),
    ]

    records = compare_image(
        image_path=Path("image.jpg"),
        ground_truth_boxes=ground_truth,
        predicted_boxes=predictions,
        inference_result=_inference_result(),
        iou_threshold=0.5,
    )

    case_types = sorted(record.case_type for record in records)
    assert case_types == ["false_negative", "false_positive", "matched"]
    false_negative = next(record for record in records if record.case_type == "false_negative")
    false_positive = next(record for record in records if record.case_type == "false_positive")
    assert false_negative.ground_truth_label == "Platelets"
    assert false_negative.yolo_label == BACKGROUND_LABEL
    assert false_positive.ground_truth_label == BACKGROUND_LABEL
    assert false_positive.yolo_label == "WBC"


def test_build_confusion_matrix_tracks_background_label() -> None:
    """Confusion matrices must keep unmatched cases through a background label."""
    records = [
        MatchRecord("image.jpg", "matched", "RBC", "RBC", "RBC", 0.9, 1.0, 0.2, False, True),
        MatchRecord("image.jpg", "false_negative", "Platelets", BACKGROUND_LABEL, BACKGROUND_LABEL, None, None, 0.2, False, True),
        MatchRecord("image.jpg", "false_positive", BACKGROUND_LABEL, "WBC", "WBC", 0.8, None, 0.2, False, True),
    ]

    labels, matrix = build_confusion_matrix(
        records,
        actual_getter=lambda record: record.ground_truth_label,
        predicted_getter=lambda record: record.yolo_label,
    )

    assert BACKGROUND_LABEL in labels
    assert matrix["RBC"]["RBC"] == 1
    assert matrix["Platelets"][BACKGROUND_LABEL] == 1
    assert matrix[BACKGROUND_LABEL]["WBC"] == 1
