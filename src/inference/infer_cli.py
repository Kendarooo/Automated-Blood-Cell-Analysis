"""CLI to run the end-to-end FR-13 inference pipeline on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.artifact_loader import ArtifactPaths
from src.inference.pipeline import InferencePipeline
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build parser for one-image end-to-end inference."""
    parser = argparse.ArgumentParser(description="Run the FR-13 inference pipeline.")
    parser.add_argument("--image", required=True, help="Microscopy image to analyze.")
    parser.add_argument(
        "--classifier-run-dir",
        required=True,
        help="Persisted ANN/SVM run directory produced by FR-11.",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/inference/baseline.json",
        help="Baseline JSON generated from the training split.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project configuration file.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run one end-to-end inference pass and print a serializable summary."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(args.classifier_run_dir)
    artifact_paths = ArtifactPaths(
        detector=Path(cfg["detection"]["best_weights"]),
        extractor_config=run_dir / "effective_config.json",
        classifier=None,
        normalizer=run_dir / "normalizer.npz",
        baseline=Path(args.baseline),
        classifier_run_dir=run_dir,
    )

    result = InferencePipeline(cfg, artifact_paths).predict(Path(args.image))
    payload = {
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
