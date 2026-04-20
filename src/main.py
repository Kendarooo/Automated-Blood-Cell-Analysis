"""Top-level pipeline orchestrator for training and inference stages."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from src.data.datasets import BCCDYoloSplitLoader
from src.detection.yolo_train import YOLOTrainer
from src.experiments.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src.experiments.run_ann_experiment import run_ann_experiment
from src.experiments.run_svm_experiment import run_svm_experiment
from src.features.extractor import ResNet18Extractor
from src.inference.artifact_loader import ArtifactPaths
from src.inference.baseline import BaselineEstimator, BaselineRepository
from src.inference.pipeline import InferencePipeline
from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed
from src.utils.wandb_logger import WandBLogger


DEFAULT_CLASS_NAMES = ("Platelets", "RBC", "WBC")


class DetectorTrainingStage:
    """Coordinate detector training without embedding model logic here."""

    def run(self, cfg: dict[str, Any]) -> dict[str, object]:
        with WandBLogger(cfg) as logger:
            best_weights = YOLOTrainer(cfg, logger=logger).train()
            run_id = logger.run_id
            run_url = logger.run_url

        cfg.setdefault("detection", {})
        cfg["detection"]["best_weights"] = str(best_weights)
        return {
            "best_weights": str(best_weights),
            "wandb_run_id": run_id,
            "wandb_run_url": run_url,
        }


class FeatureExtractionStage:
    """Prepare train/validation feature tensors and persist them for reuse."""

    def run(self, cfg: dict[str, Any]) -> dict[str, object]:
        prepared = _prepare_features(cfg)
        features_dir = Path(
            cfg.get("outputs", {}).get("features_dir", "outputs/features")
        )
        features_dir.mkdir(parents=True, exist_ok=True)

        train_path = features_dir / "train_features.npz"
        val_path = features_dir / "val_features.npz"
        metadata_path = features_dir / "feature_metadata.json"
        normalizer_path = features_dir / "normalizer.npz"

        np.savez_compressed(
            train_path,
            features=prepared.train.features,
            labels=prepared.train.labels,
        )
        np.savez_compressed(
            val_path,
            features=prepared.val.features,
            labels=prepared.val.labels,
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "class_names": list(prepared.metadata.class_names),
                    "feature_dim": prepared.metadata.feature_dim,
                    "truncate_at": prepared.metadata.truncate_at,
                    "projection_dim": prepared.metadata.projection_dim,
                    "normalization_enabled": prepared.metadata.normalization_enabled,
                    "dimensionality_strategy": prepared.metadata.dimensionality_strategy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if prepared.normalizer is not None:
            prepared.normalizer.save(str(normalizer_path))

        return {
            "prepared": prepared,
            "artifacts": {
                "train_features": str(train_path),
                "val_features": str(val_path),
                "metadata": str(metadata_path),
                "normalizer": (
                    str(normalizer_path) if prepared.normalizer is not None else None
                ),
            },
        }


class ANNTrainingStage:
    """Coordinate ANN training over already prepared features."""

    def run(self, cfg: dict[str, Any], prepared: Any) -> dict[str, object]:
        return run_ann_experiment(cfg, prepared)


class SVMTrainingStage:
    """Coordinate SVM training over already prepared features."""

    def run(self, cfg: dict[str, Any], prepared: Any) -> dict[str, object]:
        return run_svm_experiment(cfg, prepared)


class BaselineStage:
    """Estimate and persist the train-derived baseline artifact."""

    def run(self, cfg: dict[str, Any], output_path: str | None = None) -> dict[str, object]:
        dataset_cfg = cfg.get("dataset", {})
        class_names = list(dataset_cfg.get("class_names", DEFAULT_CLASS_NAMES))
        train_dir = Path(dataset_cfg.get("train_dir", "data/bccd/train"))
        resolved_output = output_path or cfg.get(
            "outputs", {}
        ).get("baseline_path", "outputs/inference/baseline.json")

        train_counts = _load_train_counts(train_dir, class_names)
        baseline = BaselineEstimator(class_names).fit_from_train_counts(train_counts, cfg)
        BaselineRepository.save(baseline, resolved_output)

        return {
            "output_path": resolved_output,
            "num_train_images": len(train_counts),
            "class_order": baseline.class_order,
            "proportions": baseline.proportions,
            "created_from_split": baseline.created_from_split,
        }


class InferenceStage:
    """Run the persisted end-to-end inference pipeline on one image."""

    def run(
        self,
        cfg: dict[str, Any],
        *,
        image_path: str,
        classifier_run_dir: str,
        baseline_path: str,
    ) -> dict[str, object]:
        run_dir = Path(classifier_run_dir)
        artifact_paths = ArtifactPaths(
            detector=Path(cfg["detection"]["best_weights"]),
            extractor_config=run_dir / "effective_config.json",
            classifier=None,
            normalizer=run_dir / "normalizer.npz",
            baseline=Path(baseline_path),
            classifier_run_dir=run_dir,
        )
        result = InferencePipeline(cfg, artifact_paths).predict(Path(image_path))
        return {
            "image_path": str(result.image_path),
            "classifier_name": result.classifier_name,
            "num_detections": result.num_detections,
            "predictions": [
                {
                    "yolo_label": prediction.yolo_label,
                    "classifier_label": prediction.classifier_label,
                    "confidence": prediction.confidence,
                }
                for prediction in result.predictions
            ],
            "statistics": {
                "counts": result.statistics.counts,
                "proportions": result.statistics.proportions,
                "total_cells": result.statistics.total_cells,
            },
            "statistical_test": {
                "p_value": result.statistical_test.p_value,
                "alert": result.statistical_test.alert,
                "test_executed": result.statistical_test.test_executed,
                "reason": result.statistical_test.reason,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the project-level CLI parser."""
    parser = argparse.ArgumentParser(description="Run the BCCD pipeline stages.")
    parser.add_argument(
        "--stage",
        choices=(
            "train_detector",
            "extract_features",
            "train_ann",
            "train_svm",
            "build_baseline",
            "infer_image",
            "all",
        ),
        required=True,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Pull hyperparameter overrides from wandb.config.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging for this invocation.",
    )
    parser.add_argument(
        "--image",
        help="Image path used by infer_image or by all when inference is requested.",
    )
    parser.add_argument(
        "--classifier-run-dir",
        help="Persisted classifier run directory used for inference.",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/inference/baseline.json",
        help="Baseline JSON artifact for inference and baseline generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run the selected pipeline stage and print its structured result."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.disable_wandb:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["enabled"] = False

    cfg, started_sweep_run = _apply_sweep_overrides(cfg, use_sweep=args.sweep)
    set_seed(int(cfg.get("seed", 42)))
    ensure_dir(cfg.get("outputs", {}).get("runs_dir", "outputs/runs"))

    result = _dispatch_stage(cfg, args)
    print(json.dumps(result, indent=2, sort_keys=True))

    if started_sweep_run:
        wandb.finish()
    return result


def _dispatch_stage(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, object]:
    detector_stage = DetectorTrainingStage()
    feature_stage = FeatureExtractionStage()
    ann_stage = ANNTrainingStage()
    svm_stage = SVMTrainingStage()
    baseline_stage = BaselineStage()
    inference_stage = InferenceStage()

    if args.stage == "train_detector":
        return detector_stage.run(cfg)

    if args.stage == "extract_features":
        extracted = feature_stage.run(cfg)
        return extracted["artifacts"]

    if args.stage == "train_ann":
        prepared = feature_stage.run(cfg)["prepared"]
        return ann_stage.run(cfg, prepared)

    if args.stage == "train_svm":
        prepared = feature_stage.run(cfg)["prepared"]
        return svm_stage.run(cfg, prepared)

    if args.stage == "build_baseline":
        return baseline_stage.run(cfg, output_path=args.baseline)

    if args.stage == "infer_image":
        classifier_run_dir = _require_argument(
            args.classifier_run_dir,
            "classifier_run_dir is required for --stage infer_image.",
        )
        image_path = _require_argument(
            args.image,
            "image is required for --stage infer_image.",
        )
        return inference_stage.run(
            cfg,
            image_path=image_path,
            classifier_run_dir=classifier_run_dir,
            baseline_path=args.baseline,
        )

    if args.stage == "all":
        detector_result = detector_stage.run(cfg)
        feature_result = feature_stage.run(cfg)
        prepared = feature_result["prepared"]
        ann_result = ann_stage.run(cfg, prepared)
        svm_result = svm_stage.run(cfg, prepared)
        baseline_result = baseline_stage.run(cfg, output_path=args.baseline)

        result: dict[str, object] = {
            "train_detector": detector_result,
            "extract_features": feature_result["artifacts"],
            "train_ann": ann_result,
            "train_svm": svm_result,
            "build_baseline": baseline_result,
        }

        if args.image is not None:
            classifier_run_dir = args.classifier_run_dir or str(ann_result["output_dir"])
            result["infer_image"] = inference_stage.run(
                cfg,
                image_path=args.image,
                classifier_run_dir=classifier_run_dir,
                baseline_path=args.baseline,
            )
        return result

    raise ValueError(f"Unsupported stage '{args.stage}'.")


def _prepare_features(cfg: dict[str, Any]):
    """Build prepared train/validation features directly from BCCD splits."""
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


def _apply_sweep_overrides(
    cfg: dict[str, Any],
    *,
    use_sweep: bool,
) -> tuple[dict[str, Any], bool]:
    """Start a sweep run and merge dotted-key overrides when requested."""
    if not use_sweep:
        return cfg, False

    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        settings=wandb.Settings(x_disable_viewer=True, silent=True),
    )
    merged_cfg = copy.deepcopy(cfg)
    _merge_flattened_overrides(merged_cfg, dict(wandb.config))
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


def _load_train_counts(
    train_dir: Path,
    class_names: list[str],
) -> list[dict[str, int]]:
    """Load per-image class counts from YOLO-format training labels."""
    labels_dir = train_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Train labels directory does not exist: {labels_dir}")

    all_counts: list[dict[str, int]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        counts = {class_name: 0 for class_name in class_names}
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            class_id = int(stripped.split()[0])
            counts[class_names[class_id]] += 1
        all_counts.append(counts)

    if not all_counts:
        raise ValueError(f"No label files found in train split: {labels_dir}")
    return all_counts


def _require_argument(value: str | None, error_message: str) -> str:
    """Return an argument value or raise a clear CLI error."""
    if value is None:
        raise ValueError(error_message)
    return value


if __name__ == "__main__":
    main()
