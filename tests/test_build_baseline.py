"""Tests for train-derived baseline generation."""

from __future__ import annotations

from pathlib import Path

from src.inference.build_baseline import _load_train_counts


def test_load_train_counts_reads_yolo_labels_per_image(tmp_path: Path) -> None:
    """Each train label file should become one per-image count dictionary."""
    labels_dir = tmp_path / "train" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("1 0.3 0.3 0.1 0.1\n2 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    counts = _load_train_counts(tmp_path / "train", ["Platelets", "RBC", "WBC"])

    assert counts == [
        {"Platelets": 1, "RBC": 1, "WBC": 0},
        {"Platelets": 0, "RBC": 1, "WBC": 1},
    ]
