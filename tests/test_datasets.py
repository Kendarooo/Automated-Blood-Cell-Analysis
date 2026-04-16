"""Tests for real BCCD split loading from YOLO-format annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.datasets import BCCDYoloSplitLoader


def _cfg() -> dict:
    return {
        "augmentation": {
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        }
    }


def _make_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labels").mkdir(parents=True)
    return split_dir


def test_bccd_yolo_split_loader_returns_cell_crops_and_labels(tmp_path: Path) -> None:
    """Each YOLO annotation must become one transformed crop and one encoded label."""
    split_dir = _make_split(tmp_path)
    image = Image.new("RGB", (100, 80), color=(255, 255, 255))
    image.save(split_dir / "images" / "sample.jpg")
    (split_dir / "labels" / "sample.txt").write_text(
        "\n".join([
            "0 0.25 0.25 0.20 0.20",
            "2 0.75 0.75 0.30 0.25",
        ]),
        encoding="utf-8",
    )

    loader = BCCDYoloSplitLoader(_cfg())
    images, labels = loader.load_split(
        split_dir,
        ("Platelets", "RBC", "WBC"),
    )

    assert images.shape == (2, 3, 224, 224)
    assert labels.dtype == np.int64
    assert labels.tolist() == [0, 2]


def test_bccd_yolo_split_loader_rejects_out_of_range_class_ids(
    tmp_path: Path,
) -> None:
    """Dataset class ids must stay aligned with the configured class order."""
    split_dir = _make_split(tmp_path)
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    image.save(split_dir / "images" / "bad.jpg")
    (split_dir / "labels" / "bad.txt").write_text(
        "3 0.50 0.50 0.25 0.25\n",
        encoding="utf-8",
    )

    loader = BCCDYoloSplitLoader(_cfg())

    with pytest.raises(ValueError, match="Invalid class_id=3"):
        loader.load_split(split_dir, ("Platelets", "RBC", "WBC"))
