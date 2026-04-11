from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Value objects  (no logic, just structured data)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box in absolute pixel coordinates.
    x1, y1: top-left corner.
    x2, y2: bottom-right corner.
    confidence: detection score.
    class_id: integer class index (0=RBC, 1=WBC, 2=Platelet or as per data.yaml).
    class_name: human-readable label.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Groups bounding boxes and their cropped sub-images for one image."""
    image_path: Path
    boxes: list[BoundingBox] = field(default_factory=list)
    crops: list[Image.Image] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cropper  (SRP: only knows how to crop)
# ---------------------------------------------------------------------------

class CellCropper:
    """
    Single Responsibility: given an image and a BoundingBox, return the crop.
    Knows nothing about YOLO or W&B.
    """

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        """
        Args:
            target_size: (width, height) to resize every crop to so that
                         downstream feature extractors receive uniform inputs
                         (satisfies NFR-6a unit test requirement).
        """
        self._target_size = target_size

    def crop(self, image: Image.Image, box: BoundingBox) -> Image.Image:
        """
        Crop and resize a single cell from image.

        Args:
            image: Full-resolution PIL image.
            box:   Bounding box in absolute pixel coords.

        Returns:
            Cropped and resized PIL image.
        """
        region = image.crop((box.x1, box.y1, box.x2, box.y2))
        return region.resize(self._target_size, Image.BILINEAR)

    def crop_all(
        self, image: Image.Image, boxes: list[BoundingBox]
    ) -> list[Image.Image]:
        """Batch-crop all detected cells from one image."""
        return [self.crop(image, box) for box in boxes]


# ---------------------------------------------------------------------------
# Inferencer  (SRP: only runs YOLO inference)
# ---------------------------------------------------------------------------

class YOLOInferencer:
    """
    Single Responsibility: loads a trained YOLO model and produces
    BoundingBox predictions for a given image.

    Dependency Inversion: receives a CellCropper via constructor so it
    can be swapped in tests without touching inference logic.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        cropper: CellCropper | None = None,
    ) -> None:
        """
        Args:
            cfg:     Full config dict returned by load_config().
            cropper: CellCropper instance (injected; defaults to standard cropper).
        """
        detection_cfg: dict[str, Any] = cfg["detection"]

        self._conf_threshold: float = detection_cfg.get("conf_threshold", 0.25)
        self._iou_threshold: float = detection_cfg.get("iou_threshold", 0.45)
        self._imgsz: int = detection_cfg["imgsz"]

        target_size = (
            detection_cfg.get("crop_width", 224),
            detection_cfg.get("crop_height", 224),
        )

        self._model = YOLO(detection_cfg["best_weights"])
        self._cropper = cropper or CellCropper(target_size=target_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, image_path: Path) -> DetectionResult:
        """
        Run YOLO inference on one image and return boxes + crops.

        Args:
            image_path: Path to the full-resolution microscopy image.

        Returns:
            DetectionResult with bounding boxes and cropped sub-images.
        """
        image = Image.open(image_path).convert("RGB")
        raw_results = self._model.predict(
            source=str(image_path),
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._imgsz,
            verbose=False,
        )

        boxes = self._parse_boxes(raw_results)
        crops = self._cropper.crop_all(image, boxes)

        return DetectionResult(image_path=image_path, boxes=boxes, crops=crops)

    def predict_batch(self, image_paths: list[Path]) -> list[DetectionResult]:
        """Run predict() over a list of image paths."""
        return [self.predict(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_boxes(self, raw_results: list) -> list[BoundingBox]:
        """
        Convert ultralytics Result objects into typed BoundingBox instances.

        Args:
            raw_results: List returned by model.predict().

        Returns:
            Flat list of BoundingBox for the single predicted image.
        """
        boxes: list[BoundingBox] = []
        result = raw_results[0]
        names: dict[int, str] = result.names

        for box in result.boxes:
            xyxy: np.ndarray = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())
            boxes.append(
                BoundingBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=float(box.conf[0].cpu().numpy()),
                    class_id=cls_id,
                    class_name=names[cls_id],
                )
            )

        return boxes