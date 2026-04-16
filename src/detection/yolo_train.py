"""Fine-tuning pipeline for YOLO object detection on the BCCD dataset."""
# src/detection/yolo_train.py
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6, Gemini.
# Description: Fine-tuning of YOLO model on BCCD dataset.
# Responsibility: Model loading, training loop, W&B metric forwarding.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils.callbacks.wb import callbacks as yolo_wb_callbacks

from src.utils.wandb_logger import WandBLogger

class YOLOModelFactory:  # pylint: disable=too-few-public-methods
    """
    Open/Closed Principle: new model variants can be added without
    modifying YOLOTrainer. Factory isolates model instantiation.
    """

    @staticmethod
    def build(weights: str) -> YOLO:
        """
        Load a YOLO model from pretrained weights.

        Args:
            weights: Path or name of pretrained weights (e.g. 'yolo26s.pt').

        Returns:
            Initialized YOLO instance.
        """
        return YOLO(weights)


class YOLOTrainer:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Single Responsibility: orchestrates YOLO fine-tuning only.
    Does NOT crop images, does NOT log to W&B directly (delegates to WandBLogger).

    Dependency Inversion: receives WandBLogger via constructor,
    never instantiates it internally.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        logger: WandBLogger,
        model_factory: YOLOModelFactory | None = None,
    ) -> None:
        """
        Args:
            cfg:           Full config dict returned by load_config().
            logger:        W&B logger instance (injected, not created here).
            model_factory: Optional factory override for testing (DIP).
        """
        self._logger = logger
        self._factory = model_factory or YOLOModelFactory()

        detection_cfg: dict[str, Any] = cfg["detection"]
        self._weights: str = detection_cfg["weights"]
        self._data_yaml: str = detection_cfg["data_yaml"]
        self._epochs: int = detection_cfg["epochs"]
        self._imgsz: int = detection_cfg["imgsz"]
        self._batch: int = detection_cfg["batch"]
        self._lr0: float = detection_cfg["lr0"]
        self._output_dir: Path = Path(detection_cfg["output_dir"])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """
        Run fine-tuning and stream per-epoch mAP to W&B.

        Returns:
            Path to the best weights saved by YOLO.
        """
        model = self._factory.build(self._weights)

        # Register W&B callbacks so YOLO streams metrics automatically.
        for event, cb in yolo_wb_callbacks.items():
            model.add_callback(event, cb)

        # Forward metrics through our WandBLogger on every epoch end.
        model.add_callback("on_fit_epoch_end", self._on_epoch_end)

        model.train(
            data=self._data_yaml,
            epochs=self._epochs,
            imgsz=self._imgsz,
            batch=self._batch,
            lr0=self._lr0,
            project=str(self._output_dir),
            name="yolo_finetune",
            exist_ok=True,
            verbose=True,
        )

        save_dir = Path(model.trainer.save_dir)
        self._log_results_csv(save_dir / "results.csv")
        best_weights = save_dir / "weights" / "best.pt"
        return best_weights

    # ------------------------------------------------------------------
    # Private callbacks
    # ------------------------------------------------------------------

    def _on_epoch_end(self, trainer: Any) -> None:  # noqa: ANN401
        """
        Callback invoked by YOLO after every epoch.
        Forwards mAP and loss metrics to W&B via the injected logger.
        """
        metrics: dict[str, float] = trainer.metrics
        epoch: int = trainer.epoch

        self._logger.log(
            {
                "epoch": epoch,
                "train/box_loss": metrics.get("train/box_loss"),
                "train/cls_loss": metrics.get("train/cls_loss"),
                "train/dfl_loss": metrics.get("train/dfl_loss"),
                "val/box_loss": metrics.get("val/box_loss"),
                "val/cls_loss": metrics.get("val/cls_loss"),
                "val/dfl_loss": metrics.get("val/dfl_loss"),
                "val/mAP50": metrics.get("metrics/mAP50"),
                "val/mAP50-95": metrics.get("metrics/mAP50-95"),
                "val/precision": metrics.get("metrics/precision"),
                "val/recall": metrics.get("metrics/recall"),
            },
            step=epoch,
        )

    def _log_results_csv(self, results_csv_path: Path) -> None:
        """
        Backfill metrics from Ultralytics' results.csv into W&B.

        The callback path can miss detection metrics depending on when
        Ultralytics populates trainer.metrics. results.csv is the most stable
        source of truth after training finishes.
        """
        if not results_csv_path.exists():
            return

        with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epoch_value = row.get("epoch")
                if epoch_value is None:
                    continue

                epoch = int(float(epoch_value)) - 1
                payload = self._build_results_payload(row)
                if payload:
                    self._logger.log(payload, step=epoch)

    @staticmethod
    def _build_results_payload(row: dict[str, str]) -> dict[str, float]:
        key_mapping = {
            "train/box_loss": "train/box_loss",
            "train/cls_loss": "train/cls_loss",
            "train/dfl_loss": "train/dfl_loss",
            "val/box_loss": "val/box_loss",
            "val/cls_loss": "val/cls_loss",
            "val/dfl_loss": "val/dfl_loss",
            "metrics/precision(B)": "metrics/precision(B)",
            "metrics/recall(B)": "metrics/recall(B)",
            "metrics/mAP50(B)": "metrics/mAP50(B)",
            "metrics/mAP50-95(B)": "metrics/mAP50-95(B)",
        }
        payload: dict[str, float] = {}

        for csv_key, log_key in key_mapping.items():
            value = row.get(csv_key)
            if value in (None, ""):
                continue
            payload[log_key] = float(value)

        return payload
