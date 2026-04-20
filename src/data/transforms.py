"""Data augmentation transforms for BCCD blood cell images (FR-4).

Design note: color alterations are intentionally conservative because
Giemsa/Wright staining encodes diagnostic information in hue.
Aggressive hue shifts would destroy the chromatic differences between
WBC (blue/purple nucleus), RBC (pink) and platelets (pale).
Geometric transforms are safe and applied more aggressively.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T


class TrainTransforms:
    """
    Single Responsibility: applies stochastic augmentation ONLY during
    training iterations (FR-4).

    Geometric: random horizontal/vertical flip, rotation, affine translation.
    Photometric: conservative brightness/contrast, mild hue jitter, gaussian noise.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Args:
            cfg: Full config dict returned by load_config().
                 Reads from cfg['augmentation'] sub-dictionary.
        """
        aug: dict[str, Any] = cfg["augmentation"]

        self._transform = T.Compose([
            T.Resize((224, 224)),
            # --- Geometric transforms (aggressive, safe for staining) ---
            T.RandomHorizontalFlip(
                p=aug.get("horizontal_flip_p", 0.5)
            ),
            T.RandomVerticalFlip(
                p=aug.get("vertical_flip_p", 0.5)
            ),
            T.RandomRotation(
                degrees=aug.get("rotation_degrees", 180)
            ),
            T.RandomAffine(
                degrees=0,
                translate=(
                    aug.get("translate_x", 0.1),
                    aug.get("translate_y", 0.1),
                ),
                scale=(
                    aug.get("scale_min", 0.9),
                    aug.get("scale_max", 1.1),
                ),
            ),
            # --- Photometric transforms (conservative, preserves staining) ---
            T.ColorJitter(
                brightness=aug.get("brightness", 0.2),
                contrast=aug.get("contrast", 0.2),
                saturation=aug.get("saturation", 0.1),
                hue=aug.get("hue", 0.05),  # very conservative: max is 0.5
            ),
            T.GaussianBlur(
                kernel_size=aug.get("blur_kernel_size", 3),
                sigma=aug.get("blur_sigma", (0.1, 0.5)),
            ),
            T.ToTensor(),
            T.Normalize(
                mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
            ),
        ])

    def __call__(self, image: Any) -> Any:
        """Apply augmentation pipeline to a PIL image."""
        return self._transform(image)


class ValTransforms:
    """
    Single Responsibility: applies ONLY deterministic preprocessing
    for validation and test sets. No stochastic augmentation (FR-4).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Args:
            cfg: Full config dict returned by load_config().
                 Reads normalize params from cfg['augmentation'].
        """
        aug: dict[str, Any] = cfg["augmentation"]

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
            ),
        ])

    def __call__(self, image: Any) -> Any:
        """Apply deterministic preprocessing to a PIL image."""
        return self._transform(image)
