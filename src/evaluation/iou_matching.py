"""Bounding-box value objects and IoU matching utilities for BCCD evaluation."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass


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


def compute_iou(box_a: GroundTruthBox, box_b: PredictedBox) -> float:
    """Compute intersection-over-union between two absolute-pixel boxes."""
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


def greedy_iou_match(
    ground_truth_boxes: list[tuple[int, GroundTruthBox]],
    predicted_boxes: list[tuple[int, PredictedBox]],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one IoU matching; caller is responsible for class filtering."""
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
