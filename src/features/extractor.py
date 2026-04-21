"""Feature extractor using a frozen pretrained ResNet18 (FR-5, CON-3, CON-4).

The convolutional backbone is truncated at a configurable intermediate layer
and its weights remain frozen throughout training (CON-4).
Only ImageNet pretrained weights are used — no hematology-specific
pretraining (CON-3).
"""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ---------------------------------------------------------------------------
# Valid truncation points and their output dimensionalities
# ---------------------------------------------------------------------------

LAYER_DIMS: dict[str, int] = {
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}


class ResNet18Extractor(nn.Module):  # pylint: disable=too-few-public-methods
    """
    Single Responsibility: maps a batch of cell images to dense feature vectors.

    The ResNet18 backbone is:
    - Pretrained on ImageNet only (CON-3).
    - Truncated at a configurable layer (FR-5).
    - Fully frozen — no gradient updates (CON-4).

    An optional trainable linear projection layer can be appended to
    compress the feature vector from K to k dimensions (FR-7).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Args:
            cfg: Full config dict returned by load_config().
                 Reads from cfg['extractor'] sub-dictionary.
        """
        super().__init__()

        extractor_cfg: dict[str, Any] = cfg["extractor"]
        self._truncate_at: str = extractor_cfg.get("truncate_at", "layer3")

        if self._truncate_at not in LAYER_DIMS:
            raise ValueError(
                f"Invalid truncate_at='{self._truncate_at}'. "
                f"Choose from {list(LAYER_DIMS.keys())}."
            )

        self._output_dim: int = LAYER_DIMS[self._truncate_at]
        projection_dim: int | None = extractor_cfg.get("projection_dim")

        # --- Build frozen backbone ---
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._backbone = self._truncate(backbone, self._truncate_at)
        self._freeze(self._backbone)

        # --- Optional trainable projection layer (FR-7) ---
        if projection_dim is not None:
            self._projection: nn.Module = nn.Linear(self._output_dim, projection_dim)
            self._final_dim: int = projection_dim
        else:
            self._projection = nn.Identity()
            self._final_dim = self._output_dim

        self._pool = nn.AdaptiveAvgPool2d((1, 1))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._final_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vector from a batch of cell images.

        Args:
            x: Tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, output_dim).
        """
        with torch.no_grad():
            features = self._backbone(x)      # (B, K, H, W)
            features = self._pool(features)   # (B, K, 1, 1)
            features = features.flatten(1)    # (B, K)

        # Projection is outside no_grad so it receives gradients (FR-7)
        return self._projection(features)     # (B, k) or (B, K)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(backbone: nn.Module, truncate_at: str) -> nn.Sequential:
        """
        Build a Sequential from the input stem up to and including
        the requested layer block.

        Args:
            backbone:    Full ResNet18 model.
            truncate_at: One of 'layer2', 'layer3', 'layer4'.

        Returns:
            Sequential containing only the layers up to truncate_at.
        """
        layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        ]

        if truncate_at in ("layer3", "layer4"):
            layers.append(backbone.layer3)

        if truncate_at == "layer4":
            layers.append(backbone.layer4)

        return nn.Sequential(*layers)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        """Freeze all parameters in a module (CON-4)."""
        for param in module.parameters():
            param.requires_grad = False
