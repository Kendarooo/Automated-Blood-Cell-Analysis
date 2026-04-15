"""Unit tests for data augmentation transforms (NFR-6c)."""

import torch
import pytest
from PIL import Image

from src.data.transforms import TrainTransforms, ValTransforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict:
    """Minimal config dict for transforms."""
    return {
        "augmentation": {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "rotation_degrees": 180,
            "translate_x": 0.1,
            "translate_y": 0.1,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.05,
            "blur_kernel_size": 3,
            "blur_sigma": [0.1, 0.5],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        }
    }


@pytest.fixture
def sample_image() -> Image.Image:
    """Synthetic RGB cell image."""
    return Image.new("RGB", (100, 80), color=(180, 120, 140))


# ---------------------------------------------------------------------------
# NFR-6c: augmentation applied only during training
# ---------------------------------------------------------------------------

class TestTransformsBehavior:
    """NFR-6c — stochastic augmentation only in TrainTransforms."""

    def test_train_transforms_returns_tensor(self, cfg, sample_image):
        """TrainTransforms must return a torch.Tensor."""
        transform = TrainTransforms(cfg)
        result = transform(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_val_transforms_returns_tensor(self, cfg, sample_image):
        """ValTransforms must return a torch.Tensor."""
        transform = ValTransforms(cfg)
        result = transform(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_train_transforms_output_shape(self, cfg, sample_image):
        """TrainTransforms must produce a (3, 224, 224) tensor."""
        transform = TrainTransforms(cfg)
        result = transform(sample_image)
        assert result.shape == (3, 224, 224), f"Expected (3,224,224), got {result.shape}"

    def test_val_transforms_output_shape(self, cfg, sample_image):
        """ValTransforms must produce a (3, 224, 224) tensor."""
        transform = ValTransforms(cfg)
        result = transform(sample_image)
        assert result.shape == (3, 224, 224), f"Expected (3,224,224), got {result.shape}"

    def test_val_transforms_are_deterministic(self, cfg, sample_image):
        """ValTransforms must produce identical output on repeated calls (no randomness)."""
        transform = ValTransforms(cfg)
        result_1 = transform(sample_image)
        result_2 = transform(sample_image)
        assert torch.allclose(result_1, result_2), (
            "ValTransforms must be deterministic but produced different outputs."
        )

    def test_train_transforms_are_stochastic(self, cfg, sample_image):
        """TrainTransforms must produce different outputs across calls (stochastic)."""
        transform = TrainTransforms(cfg)
        results = [transform(sample_image) for _ in range(20)]
        # At least one pair must differ — if all are equal, augmentation is not working
        all_equal = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_equal, (
            "TrainTransforms produced identical outputs across 20 calls — "
            "stochastic augmentation does not seem to be applied."
        )

    def test_val_does_not_contain_random_flip(self, cfg):
        """ValTransforms pipeline must not contain RandomHorizontalFlip or RandomVerticalFlip."""
        import torchvision.transforms as T
        transform = ValTransforms(cfg)
        transform_types = [type(t) for t in transform._transform.transforms]  # noqa: SLF001
        assert T.RandomHorizontalFlip not in transform_types, (
            "ValTransforms must not contain RandomHorizontalFlip."
        )
        assert T.RandomVerticalFlip not in transform_types, (
            "ValTransforms must not contain RandomVerticalFlip."
        )

    def test_val_does_not_contain_random_rotation(self, cfg):
        """ValTransforms pipeline must not contain RandomRotation."""
        import torchvision.transforms as T
        transform = ValTransforms(cfg)
        transform_types = [type(t) for t in transform._transform.transforms]  # noqa: SLF001
        assert T.RandomRotation not in transform_types, (
            "ValTransforms must not contain RandomRotation."
        )

    @pytest.mark.xfail(
        reason=(
            "ValTransforms still applies deterministic preprocessing "
            "(Resize/ToTensor/Normalize), so the output is not pixel-identical "
            "to the input image."
        ),
        strict=True,
    )
    def test_eval_output_is_identical_to_input(self):
        """
        Point 3 requested check:
        in eval mode the output should be identical to the input image.

        This test currently documents a gap in the implementation:
        validation/inference avoids stochastic augmentation, but it still
        applies deterministic preprocessing that changes pixel values.
        """
        import torchvision.transforms as T

        cfg = {
            "augmentation": {
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            }
        }

        sample_image = Image.new("RGB", (224, 224), color=(180, 120, 140))
        transform = ValTransforms(cfg)

        output = transform(sample_image)
        reference = T.ToTensor()(sample_image)

        assert torch.allclose(output, reference), (
            "Eval output should be identical to input, but deterministic "
            "preprocessing still changes the image."
        )
