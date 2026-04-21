"""Tests for persisted YOLO run restoration in FR-13."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.run_loader import load_yolo_run


def test_load_yolo_run_restores_weights_and_config(monkeypatch, tmp_path: Path) -> None:
    """A persisted YOLO run must restore its weights and exact config without retraining."""
    run_dir = tmp_path / "yolo_finetune"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_weights = weights_dir / "best.pt"
    best_weights.write_text("stub-weights", encoding="utf-8")

    config = {
        "detection": {
            "weights": "yolo26s.pt",
            "best_weights": str(best_weights),
            "imgsz": 320,
        }
    }
    summary = {
        "detector_name": "yolo",
        "local_artifact_paths": {
            "weights": str(best_weights),
            "config": str(run_dir / "effective_config.json"),
        },
    }
    (run_dir / "effective_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    captured: dict[str, str] = {}

    class StubYOLO:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

    monkeypatch.setattr("src.detection.run_loader.YOLO", StubYOLO)

    loaded = load_yolo_run(run_dir)

    assert captured["weights"] == str(best_weights)
    assert loaded.best_weights == best_weights
    assert loaded.effective_config["detection"]["weights"] == "yolo26s.pt"


def test_load_yolo_run_requires_summary_file(tmp_path: Path) -> None:
    """Missing YOLO metadata must fail before any partial restoration."""
    with pytest.raises(FileNotFoundError, match="run_summary.json"):
        load_yolo_run(tmp_path / "missing_yolo_run")
