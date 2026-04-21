"""Tests for YOLO training metric forwarding."""

from __future__ import annotations

import json
from pathlib import Path

from src.detection.yolo_train import YOLOTrainer


class RecordingLogger:
    """Collect metric payloads forwarded by YOLOTrainer."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int | None]] = []

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.calls.append((metrics, step))


def _cfg(tmp_path: Path) -> dict:
    return {
        "detection": {
            "weights": "yolo26s.pt",
            "data_yaml": "data/bccd/data.yaml",
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "lr0": 0.01,
            "output_dir": str(tmp_path / "outputs"),
        }
    }


def test_yolo_trainer_logs_metrics_from_results_csv(tmp_path: Path) -> None:
    """Final YOLO metrics must be backfilled from results.csv into W&B."""
    logger = RecordingLogger()
    trainer = YOLOTrainer(cfg=_cfg(tmp_path), logger=logger)
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        (
            "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
            "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
            "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
            "1,13.99,4.06735,3.29601,0.07948,0.8625,0.21102,0.19069,0.10304,"
            "3.46404,2.9867,0.04905\n"
        ),
        encoding="utf-8",
    )

    trainer._log_results_csv(results_csv)  # noqa: SLF001

    assert len(logger.calls) == 1
    payload, step = logger.calls[0]
    assert step == 0
    assert payload["train/box_loss"] == 4.06735
    assert payload["metrics/precision(B)"] == 0.8625
    assert payload["metrics/mAP50(B)"] == 0.19069
    assert payload["val/dfl_loss"] == 0.04905


class StubYOLOModel:
    """Minimal YOLO stub for persistence-oriented trainer tests."""

    def __init__(self, save_dir: Path) -> None:
        self._save_dir = save_dir
        self.trainer = type("Trainer", (), {"save_dir": str(save_dir)})()

    def add_callback(self, event: str, callback) -> None:  # noqa: ANN001
        del event, callback

    def train(self, **kwargs) -> None:  # noqa: ANN003
        del kwargs
        weights_dir = self._save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_text("stub-weights", encoding="utf-8")
        (self._save_dir / "results.csv").write_text("epoch\n", encoding="utf-8")


class StubYOLOFactory:
    """Factory stub returning a deterministic YOLO-like object."""

    def __init__(self, save_dir: Path) -> None:
        self._save_dir = save_dir

    def build(self, weights: str) -> StubYOLOModel:
        del weights
        return StubYOLOModel(self._save_dir)


def test_yolo_trainer_persists_detector_config_and_summary(tmp_path: Path) -> None:
    """Training must persist detector weights metadata and exact config for restoration."""
    logger = RecordingLogger()
    save_dir = tmp_path / "outputs" / "yolo_finetune"
    trainer = YOLOTrainer(
        cfg=_cfg(tmp_path),
        logger=logger,
        model_factory=StubYOLOFactory(save_dir),
    )

    best_weights = trainer.train()

    summary_path = save_dir / "run_summary.json"
    config_path = save_dir / "effective_config.json"

    assert best_weights == save_dir / "weights" / "best.pt"
    assert summary_path.exists()
    assert config_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    effective_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert summary["detector_name"] == "yolo"
    assert summary["local_artifact_paths"]["weights"] == str(best_weights)
    assert effective_config["detection"]["weights"] == "yolo26s.pt"
