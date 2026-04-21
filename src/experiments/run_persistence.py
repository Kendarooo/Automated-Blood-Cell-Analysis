"""Local persistence utilities for FR-11 experiment runs."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def build_run_id(classifier_name: str) -> str:
    """Return a short unique identifier for one experiment run."""
    return f"{classifier_name}_{uuid.uuid4().hex[:8]}"


def persist_run(
    *,
    run_id: str,
    classifier_name: str,
    output_root: str | Path,
    effective_config: dict[str, Any],
    metrics: dict[str, Any],
    feature_metadata: dict[str, Any],
    model_path: str | Path,
    normalizer_path: str | Path | None = None,
    metric_name: str = "val_macro_f1",
    sweep_phase: str | None = None,
    seed: int | None = None,
    wandb_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist JSON summaries for one run and return path metadata."""
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "effective_config.json"
    feature_metadata_path = output_dir / "feature_metadata.json"
    summary_path = output_dir / "run_summary.json"
    resolved_model_path = Path(model_path)
    resolved_normalizer_path = Path(normalizer_path) if normalizer_path is not None else None

    _write_json(metrics_path, metrics)
    _write_json(config_path, effective_config)
    _write_json(feature_metadata_path, feature_metadata)

    summary = {
        "run_id": run_id,
        "classifier_name": classifier_name,
        "sweep_phase": sweep_phase or classifier_name,
        "metric_name": metric_name,
        "metric_value": metrics[metric_name],
        "status": "completed",
        "seed": seed,
        "feature_metadata": feature_metadata,
        "metrics": metrics,
        "effective_config": effective_config,
        "local_artifact_paths": {
            "model": str(resolved_model_path),
            "metrics": str(metrics_path),
            "config": str(config_path),
            "feature_metadata": str(feature_metadata_path),
        },
        "wandb": wandb_metadata or {"enabled": False},
    }
    if resolved_normalizer_path is not None:
        summary["local_artifact_paths"]["normalizer"] = str(resolved_normalizer_path)
    _write_json(summary_path, summary)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "run_summary_path": str(summary_path),
        "local_artifact_paths": summary["local_artifact_paths"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
