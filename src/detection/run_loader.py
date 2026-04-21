"""Restoration helpers for persisted YOLO detector runs."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass(frozen=True)
class LoadedYOLOArtifacts:
    """Restored YOLO weights plus the exact config that produced them."""

    model: YOLO
    effective_config: dict[str, Any]
    run_summary: dict[str, Any]
    run_dir: Path
    best_weights: Path


def load_yolo_run(run_dir: str | Path) -> LoadedYOLOArtifacts:
    """Restore one persisted YOLO run from disk without retraining."""
    resolved_run_dir = Path(run_dir)
    summary_path = resolved_run_dir / "run_summary.json"
    config_path = resolved_run_dir / "effective_config.json"

    _require_existing_path(summary_path)
    _require_existing_path(config_path)

    run_summary = _read_json(summary_path)
    effective_config = _read_json(config_path)
    best_weights = Path(run_summary["local_artifact_paths"]["weights"])
    _require_existing_path(best_weights)

    return LoadedYOLOArtifacts(
        model=YOLO(str(best_weights)),
        effective_config=effective_config,
        run_summary=run_summary,
        run_dir=resolved_run_dir,
        best_weights=best_weights,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_existing_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"YOLO artifact path does not exist: {path}")
