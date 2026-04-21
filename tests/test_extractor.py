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

# ---------------------------------------------------------------------------
# NFR-6b: extractor produces vectors of expected dimensionality
# ---------------------------------------------------------------------------

class TestResNet18ExtractorDimensionality:
    """NFR-6b — extractor output matches the configured truncation layer."""

    @pytest.fixture
    def base_cfg(self) -> dict:
        """Minimal config dict for the extractor."""
        return {
            "augmentation": {
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            }
        }

    def _make_cfg(self, base_cfg, truncate_at, projection_dim=None):
        base_cfg["extractor"] = {
            "truncate_at": truncate_at,
            "projection_dim": projection_dim,
        }
        return base_cfg

    def test_layer2_produces_128_dims(self, base_cfg):
        """Truncating at layer2 must produce 128-dimensional vectors."""
        import torch
        from src.features.extractor import ResNet18Extractor
        extractor = ResNet18Extractor(self._make_cfg(base_cfg, "layer2"))
        output = extractor(torch.zeros(2, 3, 224, 224))
        assert output.shape == (2, 128)
        assert extractor.output_dim == 128

    def test_layer3_produces_256_dims(self, base_cfg):
        """Truncating at layer3 must produce 256-dimensional vectors."""
        import torch
        from src.features.extractor import ResNet18Extractor
        extractor = ResNet18Extractor(self._make_cfg(base_cfg, "layer3"))
        output = extractor(torch.zeros(2, 3, 224, 224))
        assert output.shape == (2, 256)
        assert extractor.output_dim == 256

    def test_layer4_produces_512_dims(self, base_cfg):
        """Truncating at layer4 must produce 512-dimensional vectors."""
        import torch
        from src.features.extractor import ResNet18Extractor
        extractor = ResNet18Extractor(self._make_cfg(base_cfg, "layer4"))
        output = extractor(torch.zeros(2, 3, 224, 224))
        assert output.shape == (2, 512)
        assert extractor.output_dim == 512

    def test_projection_reduces_dimensionality(self, base_cfg):
        """Projection layer must compress output to projection_dim."""
        import torch
        from src.features.extractor import ResNet18Extractor
        extractor = ResNet18Extractor(self._make_cfg(base_cfg, "layer4", projection_dim=64))
        output = extractor(torch.zeros(2, 3, 224, 224))
        assert output.shape == (2, 64)
        assert extractor.output_dim == 64

    def test_invalid_layer_raises(self, base_cfg):
        """An invalid truncate_at value must raise ValueError immediately."""
        from src.features.extractor import ResNet18Extractor
        with pytest.raises(ValueError):
            ResNet18Extractor(self._make_cfg(base_cfg, "layer99"))

    def test_backbone_parameters_are_frozen(self, base_cfg):
        """All backbone parameters must have requires_grad=False (CON-4)."""
        from src.features.extractor import ResNet18Extractor
        extractor = ResNet18Extractor(self._make_cfg(base_cfg, "layer3"))
        for param in extractor._backbone.parameters():  # noqa: SLF001
            assert not param.requires_grad


class TestCON4GradientIsolation:
    """CON-4 — backbone grads are None after backward; projection grads are not."""

    @pytest.fixture
    def extractor_with_projection(self):
        import torch
        from src.features.extractor import ResNet18Extractor
        cfg = {
            "extractor": {"truncate_at": "layer3", "projection_dim": 64},
            "augmentation": {
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }
        extractor = ResNet18Extractor(cfg)
        extractor.train()
        return extractor

    def test_backbone_grads_are_none_after_backward(self, extractor_with_projection):
        """After a full forward+backward pass, backbone params must have grad=None."""
        import torch
        extractor = extractor_with_projection
        x = torch.randn(2, 3, 224, 224)
        loss = extractor(x).sum()
        loss.backward()
        for param in extractor._backbone.parameters():  # noqa: SLF001
            assert param.grad is None, (
                f"Backbone param {param.shape} has gradients — CON-4 violated."
            )

    def test_projection_grads_are_not_none_after_backward(self, extractor_with_projection):
        """After a full forward+backward pass, projection params must have non-None grad."""
        import torch
        extractor = extractor_with_projection
        x = torch.randn(2, 3, 224, 224)
        loss = extractor(x).sum()
        loss.backward()
        projection_params = list(extractor._projection.parameters())  # noqa: SLF001
        assert projection_params, "Projection layer has no parameters."
        for param in projection_params:
            assert param.grad is not None, (
                f"Projection param {param.shape} has no gradient after backward."
            )
