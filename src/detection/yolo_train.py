"""Fine-tuning pipeline for YOLO object detection on the BCCD dataset."""
# src/detection/yolo_train.py
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6, Gemini.
# Description: Fine-tuning of YOLO model on BCCD dataset.
# Responsibility: Model loading, training loop, W&B metric forwarding.

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

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
        self._cfg = deepcopy(cfg)

        detection_cfg: dict[str, Any] = cfg["detection"]
        self._weights: str = detection_cfg["weights"]
        self._data_yaml: str = detection_cfg["data_yaml"]
        self._epochs: int = detection_cfg["epochs"]
        self._imgsz: int = detection_cfg["imgsz"]
        self._batch: int = detection_cfg["batch"]
        self._lr0: float = detection_cfg["lr0"]
        self._output_dir: Path = Path(detection_cfg["output_dir"])
        self._yolo_model: YOLO | None = None  # set in train(), used in _on_epoch_end

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
        self._yolo_model = model

        # Register W&B callbacks so YOLO streams metrics automatically.
        for event, cb in yolo_wb_callbacks.items():
            model.add_callback(event, cb)

        # Forward metrics through our WandBLogger on every epoch end.
        model.add_callback("on_fit_epoch_end", self._on_epoch_end)

        # Use a train-only YAML so YOLO's internal loop never touches
        # valid/ or test/ images — actual val mAP is computed explicitly
        # in _on_epoch_end using the original data yaml.
        train_only_yaml = self._make_train_only_yaml()
        try:
            model.train(
                data=str(train_only_yaml),
                epochs=self._epochs,
                imgsz=self._imgsz,
                batch=self._batch,
                lr0=self._lr0,
                project=str(self._output_dir),
                name="yolo_finetune",
                exist_ok=True,
                verbose=True,
            )
        finally:
            train_only_yaml.unlink(missing_ok=True)

        save_dir = Path(model.trainer.save_dir)
        self._log_results_csv(save_dir / "results.csv")
        best_weights = save_dir / "weights" / "best.pt"
        self._persist_run_artifacts(save_dir=save_dir, best_weights=best_weights)
        return best_weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_train_only_yaml(self) -> Path:
        """Write a sibling YAML that exposes only train/images to YOLO.

        Ultralytics requires a 'val' key; we deliberately point it to the
        same train path so YOLO's internal validation loop never loads
        held-out images.  Actual val-split mAP is evaluated explicitly in
        _on_epoch_end using self._data_yaml (the original full YAML).
        """
        original = yaml.safe_load(Path(self._data_yaml).read_text(encoding="utf-8"))
        train_path = original["train"]
        train_only = {
            "train": train_path,
            "val":   train_path,   # intentionally points to train only
            "nc":    original["nc"],
            "names": original["names"],
        }
        out_path = Path(self._data_yaml).parent / "_train_only.yaml"
        out_path.write_text(yaml.dump(train_only), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # Private callbacks
    # ------------------------------------------------------------------

    def _on_epoch_end(self, trainer: Any) -> None:  # noqa: ANN401
        """Callback invoked by YOLO after every epoch.

        Train losses come from trainer.metrics (computed on train split).
        Val mAP is obtained by running an explicit evaluation pass on the
        held-out valid/ split via self._data_yaml — guaranteeing that no
        validation image is ever seen during the weight-update phase.
        """
        epoch: int = trainer.epoch
        metrics: dict[str, float] = trainer.metrics

        # Explicit evaluation on the held-out val split.
        # YOLO.val() sets the model to eval mode and uses torch.no_grad().
        val_results = self._yolo_model.val(  # type: ignore[union-attr]
            data=self._data_yaml,
            split="val",
            verbose=False,
            plots=False,
        )

        self._logger.log(
            {
                "epoch": epoch,
                "train/box_loss": metrics.get("train/box_loss"),
                "train/cls_loss": metrics.get("train/cls_loss"),
                "train/dfl_loss": metrics.get("train/dfl_loss"),
                "val/mAP50":      val_results.box.map50,
                "val/mAP50-95":   val_results.box.map,
                "val/precision":  val_results.box.mp,
                "val/recall":     val_results.box.mr,
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

    def _persist_run_artifacts(self, *, save_dir: Path, best_weights: Path) -> None:
        """Persist the exact detector config next to the trained YOLO weights."""
        config_path = save_dir / "effective_config.json"
        summary_path = save_dir / "run_summary.json"
        payload = {
            "detector_name": "yolo",
            "best_weights": str(best_weights),
            "local_artifact_paths": {
                "weights": str(best_weights),
                "config": str(config_path),
            },
            "effective_config": self._cfg,
        }
        config_path.write_text(
            json.dumps(self._cfg, indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
