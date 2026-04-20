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
        self._enabled = bool(wandb_cfg.get("enabled", True))

        self._run = None
        self._owns_run = False
        if self._enabled:
            if wandb.run is not None:
                self._run = wandb.run
            else:
                self._run = wandb.init(
                    project=wandb_cfg["project"],
                    entity=wandb_cfg.get("entity"),
                    name=run_name or wandb_cfg.get("run_name"),
                    config=cfg,
                    reinit=True,
                    settings=wandb.Settings(x_disable_viewer=True, silent=True),
                )
                self._owns_run = True
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics for a given step/epoch."""
        if self._run is not None:
            self._run.log(metrics, step=step)
    def log_confusion_matrix(
        self,
        labels: list[str],
        y_true: list[int],
        preds: list[int],
    ) -> None:
        """Log a confusion matrix as a W&B artifact."""
        if self._run is not None:
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
        if self._run is not None and self._owns_run:
            self._run.finish()

    @property
    def run_id(self) -> str | None:
        """Return the active W&B run id when logging is enabled."""
        return None if self._run is None else self._run.id

    @property
    def run_url(self) -> str | None:
        """Return the active W&B run URL when logging is enabled."""
        return None if self._run is None else self._run.url
    # ------------------------------------------------------------------
    # Context manager  (with WandBLogger(cfg) as logger:)
    # ------------------------------------------------------------------
    def __enter__(self) -> "WandBLogger":
        return self
    def __exit__(self, *_: Any) -> None:
        self.finish()
