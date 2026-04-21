"""CLI entrypoint for FR-11 experiment runners."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import wandb

from src.data.datasets import BCCDYoloSplitLoader
from src.experiments.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src.experiments.run_ann_experiment import run_ann_experiment
from src.experiments.run_svm_experiment import run_svm_experiment
from src.features.extractor import ResNet18Extractor
from src.utils.config import load_config
from src.utils.seed import set_seed


DEFAULT_CLASS_NAMES = ("Platelets", "RBC", "WBC")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for manual experiment execution."""
    parser = argparse.ArgumentParser(description="Run FR-11 ANN or SVM experiments.")
    parser.add_argument(
        "--runner",
        choices=("ann", "svm"),
        required=True,
        help="Classifier runner to execute.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the base project config file.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Force-disable W&B logging for this invocation.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Execute one experiment runner from the command line."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.disable_wandb:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["enabled"] = False

    cfg, cli_started_wandb = _apply_wandb_overrides(cfg)
    set_seed(int(cfg.get("seed", 42)))

    prepared = _prepare_features(cfg)
    result = _dispatch_runner(args.runner, cfg, prepared)
    print(json.dumps(result, indent=2, sort_keys=True))
    if cli_started_wandb:
        wandb.finish()
    return result


def _apply_wandb_overrides(cfg: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Merge sweep overrides from W&B when running under a sweep agent."""
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))
    if not use_wandb or "WANDB_SWEEP_ID" not in os.environ:
        return cfg, False

    wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        settings=wandb.Settings(x_disable_viewer=True, silent=True),
    )
    merged_cfg = copy.deepcopy(cfg)
    overrides = dict(wandb.config)
    _merge_flattened_overrides(merged_cfg, overrides)
    return merged_cfg, True


def _merge_flattened_overrides(
    cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply flattened dotted-key overrides onto a nested config dictionary."""
    for dotted_key, value in overrides.items():
        _set_nested_value(cfg, dotted_key, value)
    return cfg


def _set_nested_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign one dotted-path value inside a nested dict, creating levels as needed."""
    path = dotted_key.split(".")
    cursor = cfg
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _prepare_features(cfg: dict[str, Any]):
    """Build prepared train/validation features directly from BCCD YOLO splits."""
    dataset_cfg = cfg.get("dataset", {})
    train_dir = Path(dataset_cfg.get("train_dir", "data/bccd/train"))
    val_dir = Path(dataset_cfg.get("val_dir", "data/bccd/valid"))
    class_names = tuple(dataset_cfg.get("class_names", DEFAULT_CLASS_NAMES))
    extractor_cfg = cfg.get("extractor", {})
    projection_dim = extractor_cfg.get("projection_dim")
    dimensionality_strategy = "projection" if projection_dim is not None else "none"

    pipeline = FeaturePipeline(
        config=FeaturePipelineConfig(
            train_dir=train_dir,
            val_dir=val_dir,
            truncate_at=extractor_cfg.get("truncate_at", "layer3"),
            projection_dim=projection_dim,
            normalization_enabled=bool(
                cfg.get("feature_pipeline", {}).get("normalization_enabled", True)
            ),
            dimensionality_strategy=cfg.get("feature_pipeline", {}).get(
                "dimensionality_strategy",
                dimensionality_strategy,
            ),
            class_names=class_names,
            extraction_batch_size=int(dataset_cfg.get("batch_size", 32)),
        ),
        split_loader=BCCDYoloSplitLoader(cfg),
        extractor=ResNet18Extractor(cfg),
    )
    return pipeline.prepare_train_val_features()


def _dispatch_runner(
    runner_name: str,
    cfg: dict[str, Any],
    prepared: Any,
) -> dict[str, object]:
    if runner_name == "ann":
        return run_ann_experiment(cfg, prepared)
    if runner_name == "svm":
        return run_svm_experiment(cfg, prepared)
    raise ValueError(f"Unsupported runner '{runner_name}'.")


if __name__ == "__main__":
    main()
