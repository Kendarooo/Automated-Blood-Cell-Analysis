"""Centralized Weights & Biases telemetry logger for the BCCD pipeline."""
from __future__ import annotations
from typing import Any
import wandb
class WandBLogger:
    """
    Single Responsibility: handles all interaction with Weights & Biases.
    No training logic, no file I/O, no model loading lives here.
    """
    def __init__(self, cfg: dict[str, Any], run_name: str | None = None) -> None:
        """
        Args:
            cfg:      Full config dict returned by load_config().
            run_name: Optional override for the run name.
        """
        wandb_cfg: dict[str, Any] = cfg["wandb"]

        self._run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=run_name or wandb_cfg.get("run_name"),
            config=cfg,
            reinit=True,
        )
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics for a given step/epoch."""
        self._run.log(metrics, step=step)
    def log_confusion_matrix(
        self,
        labels: list[str],
        y_true: list[int],
        preds: list[int],
    ) -> None:
        """Log a confusion matrix as a W&B artifact."""
        self._run.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=preds,
                    class_names=labels,
                )
            }
        )
    def finish(self) -> None:
        """Close the W&B run cleanly."""
        self._run.finish()
    # ------------------------------------------------------------------
    # Context manager  (with WandBLogger(cfg) as logger:)
    # ------------------------------------------------------------------
    def __enter__(self) -> "WandBLogger":
        return self
    def __exit__(self, *_: Any) -> None:
        self.finish()
