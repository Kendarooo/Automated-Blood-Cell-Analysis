"""Tests for YOLO training metric forwarding."""

from __future__ import annotations

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
