"""Unit tests for the detection module - covers NFR-6a requirement."""

import pytest
from PIL import Image

from src.detection.yolo_infer import BoundingBox, CellCropper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def full_image() -> Image.Image:
    """Synthetic full-resolution microscopy image (640x480 RGB)."""
    return Image.new("RGB", (640, 480), color=(200, 180, 160))


@pytest.fixture
def cropper() -> CellCropper:
    """CellCropper with default target size 224x224."""
    return CellCropper(target_size=(224, 224))


def make_box(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Helper to build a BoundingBox without worrying about score/class."""
    return BoundingBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=0.9,
        class_id=0,
        class_name="RBC",
    )


# ---------------------------------------------------------------------------
# NFR-6a: all crops must have identical spatial dimensions
# regardless of the original bounding box size
# ---------------------------------------------------------------------------

class TestCellCropperUniformSize:
    """NFR-6a — crops delivered to the extractor have identical dimensions."""

    def test_small_box_produces_target_size(self, full_image, cropper):
        """A tiny bounding box must still produce a 224x224 crop."""
        box = make_box(10, 10, 30, 30)  # 20x20 px box
        crop = cropper.crop(full_image, box)
        assert crop.size == (224, 224), (
            f"Expected (224, 224) but got {crop.size} for a small box."
        )

    def test_large_box_produces_target_size(self, full_image, cropper):
        """A large bounding box must also produce a 224x224 crop."""
        box = make_box(0, 0, 400, 300)  # 400x300 px box
        crop = cropper.crop(full_image, box)
        assert crop.size == (224, 224), (
            f"Expected (224, 224) but got {crop.size} for a large box."
        )

    def test_non_square_box_produces_target_size(self, full_image, cropper):
        """A non-square bounding box must produce a square 224x224 crop."""
        box = make_box(50, 50, 200, 100)  # 150x50 px — wide rectangle
        crop = cropper.crop(full_image, box)
        assert crop.size == (224, 224), (
            f"Expected (224, 224) but got {crop.size} for a non-square box."
        )

    def test_all_crops_same_size(self, full_image, cropper):
        """
        NFR-6a core: multiple boxes of different sizes must all
        produce crops with identical dimensions.
        """
        boxes = [
            make_box(0, 0, 20, 20),
            make_box(100, 100, 250, 180),
            make_box(300, 200, 600, 450),
        ]
        crops = cropper.crop_all(full_image, boxes)
        sizes = {crop.size for crop in crops}
        assert len(sizes) == 1, (
            f"Expected all crops to be the same size, got multiple: {sizes}"
        )
        assert sizes.pop() == (224, 224)

    def test_custom_target_size_respected(self, full_image):
        """A CellCropper with a custom target size must honor it."""
        custom_cropper = CellCropper(target_size=(128, 128))
        box = make_box(10, 10, 100, 100)
        crop = custom_cropper.crop(full_image, box)
        assert crop.size == (128, 128)

    def test_empty_box_list_returns_empty(self, full_image, cropper):
        """crop_all with no boxes must return an empty list without errors."""
        crops = cropper.crop_all(full_image, [])
        assert crops == []

    def test_crop_returns_rgb_image(self, full_image, cropper):
        """Crops must be RGB PIL images, not grayscale or RGBA."""
        box = make_box(10, 10, 80, 80)
        crop = cropper.crop(full_image, box)
        assert crop.mode == "RGB"