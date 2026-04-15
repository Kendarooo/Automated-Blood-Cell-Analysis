"""FR-5 tests for configurable truncation in the feature extractor."""

import torch

from src.features.extractor import ResNet18Extractor


def _cfg(truncate_at: str) -> dict:
    """Build a minimal extractor config."""
    return {
        "extractor": {
            "truncate_at": truncate_at,
            "projection_dim": None,
        }
    }


def test_fr5_layer2_outputs_128_dims() -> None:
    """layer2 truncation must produce 128-dimensional feature vectors."""
    extractor = ResNet18Extractor(_cfg("layer2"))
    output = extractor(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 128)


def test_fr5_layer3_outputs_256_dims() -> None:
    """layer3 truncation must produce 256-dimensional feature vectors."""
    extractor = ResNet18Extractor(_cfg("layer3"))
    output = extractor(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 256)


def test_fr5_layer4_outputs_512_dims() -> None:
    """layer4 truncation must produce 512-dimensional feature vectors."""
    extractor = ResNet18Extractor(_cfg("layer4"))
    output = extractor(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 512)

