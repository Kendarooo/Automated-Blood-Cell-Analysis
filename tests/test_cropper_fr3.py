"""FR-3 unit tests for cellular cropping from YOLO bounding boxes."""

from PIL import Image

from src.detection.yolo_infer import BoundingBox, CellCropper


def _box(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Build a test bounding box with dummy metadata."""
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=0.9,
        class_id=0,
        class_name="RBC",
    )


def test_fr3_cropper_resizes_all_boxes_to_fixed_size() -> None:
    """
    FR-3:
    The cropper must take YOLO-style bounding boxes, crop from the original
    image, and resize every subimage to the same spatial dimensions.
    """
    image = Image.new("RGB", (416, 416), color=(180, 160, 140))
    boxes = [
        _box(10, 20, 70, 90),      # small, almost square
        _box(100, 40, 260, 120),   # wide rectangle
        _box(280, 150, 340, 360),  # tall rectangle
    ]

    cropper = CellCropper(target_size=(224, 224))
    crops = cropper.crop_all(image, boxes)

    assert len(crops) == 3
    assert all(crop.size == (224, 224) for crop in crops)

