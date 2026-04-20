"""Run restoration utilities for persisted FR-13 experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from src.models.ann import ANNFactory, BloodCellANN


@dataclass(frozen=True)
class LoadedRunArtifacts:
    """Restored model plus persisted metadata for one experiment run."""

    classifier_name: str
    model: object
    effective_config: dict[str, Any]
    feature_metadata: dict[str, Any]
    run_summary: dict[str, Any]
    run_dir: Path


def load_run(run_dir: str | Path) -> LoadedRunArtifacts:
    """Restore one persisted ANN or SVM run from disk."""
    resolved_run_dir = Path(run_dir)
    summary_path = resolved_run_dir / "run_summary.json"
    config_path = resolved_run_dir / "effective_config.json"
    feature_metadata_path = resolved_run_dir / "feature_metadata.json"

    _require_existing_path(summary_path)
    _require_existing_path(config_path)
    _require_existing_path(feature_metadata_path)

    run_summary = _read_json(summary_path)
    effective_config = _read_json(config_path)
    feature_metadata = _read_json(feature_metadata_path)
    classifier_name = str(run_summary["classifier_name"])
    model_path = Path(run_summary["local_artifact_paths"]["model"])
    _require_existing_path(model_path)

    if classifier_name == "ann":
        model = _load_ann_model(
            effective_config=effective_config,
            feature_metadata=feature_metadata,
            model_path=model_path,
        )
    elif classifier_name == "svm":
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported classifier_name '{classifier_name}'.")

    return LoadedRunArtifacts(
        classifier_name=classifier_name,
        model=model,
        effective_config=effective_config,
        feature_metadata=feature_metadata,
        run_summary=run_summary,
        run_dir=resolved_run_dir,
    )


def _load_ann_model(
    *,
    effective_config: dict[str, Any],
    feature_metadata: dict[str, Any],
    model_path: Path,
) -> BloodCellANN:
    input_dim = int(feature_metadata["feature_dim"])
    model = ANNFactory.from_project_config(effective_config, input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_existing_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Run artifact path does not exist: {path}")
