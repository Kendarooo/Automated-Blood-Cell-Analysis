"""Batch inference over a secondary synthetic image set for FR-14-style reporting."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.inference.artifact_loader import ArtifactPaths
from src.inference.pipeline import InferencePipeline
from src.utils.config import load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    """Build parser for batch inference over a synthetic image directory."""
    parser = argparse.ArgumentParser(
        description="Run the trained pipeline over a synthetic image set.",
    )
    parser.add_argument(
        "--synthetic-dir",
        required=True,
        help="Directory containing the synthetic microscopy images.",
    )
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
    parser.add_argument(
        "--output-dir",
        default="outputs/inference/synthetic_set",
        help="Directory where the CSV/JSON summaries will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run batch inference over a synthetic directory and persist per-image summaries."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(args.classifier_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = ArtifactPaths(
        detector=Path(cfg["detection"]["best_weights"]),
        extractor_config=run_dir / "effective_config.json",
        classifier=None,
        normalizer=run_dir / "normalizer.npz",
        baseline=Path(args.baseline),
        classifier_run_dir=run_dir,
    )
    pipeline = InferencePipeline(cfg, artifact_paths)

    image_paths = _collect_image_paths(Path(args.synthetic_dir))
    rows = [_summarize_prediction(pipeline.predict(image_path)) for image_path in image_paths]

    csv_path = output_dir / "synthetic_results.csv"
    _write_results_csv(csv_path, rows)

    summary = {
        "synthetic_dir": str(Path(args.synthetic_dir)),
        "classifier_run_dir": str(run_dir),
        "num_images": len(rows),
        "num_alerts": sum(bool(row["alert"]) for row in rows),
        "results_csv": str(csv_path),
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _collect_image_paths(synthetic_dir: Path) -> list[Path]:
    """Return sorted image paths from a flat or nested synthetic image directory."""
    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Synthetic directory does not exist: {synthetic_dir}")

    image_paths = sorted(
        path
        for path in synthetic_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images were found in synthetic directory: {synthetic_dir}"
        )
    return image_paths


def _summarize_prediction(result) -> dict[str, object]:
    """Convert one inference result into a CSV-friendly per-image summary row."""
    proportions = result.statistics.proportions
    return {
        "image_path": str(result.image_path),
        "classifier_name": result.classifier_name,
        "num_detections": result.num_detections,
        "wbc_pct": round(float(proportions.get("WBC", 0.0) * 100.0), 6),
        "rbc_pct": round(float(proportions.get("RBC", 0.0) * 100.0), 6),
        "plt_pct": round(float(proportions.get("Platelets", 0.0) * 100.0), 6),
        "p_value": result.statistical_test.p_value,
        "alert": bool(result.statistical_test.alert),
        "test_executed": bool(result.statistical_test.test_executed),
    }


def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist one CSV row per synthetic image."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "classifier_name",
                "num_detections",
                "wbc_pct",
                "rbc_pct",
                "plt_pct",
                "p_value",
                "alert",
                "test_executed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
