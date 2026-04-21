"""Calibrate YOLO confidence threshold on BCCD validation data."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.compare_labels import main as compare_labels_main

"""Configura los parámetros de la terminal para definir los directorios de datos y la 
lista de umbrales de confianza (thresholds) a evaluar."""
def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for threshold calibration over validation data."""
    parser = argparse.ArgumentParser(
        description="Calibrate YOLO confidence threshold on a labeled BCCD split.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project configuration file.",
    )
    parser.add_argument(
        "--classifier-run-dir",
        required=True,
        help="Persisted ANN/SVM run directory used by the inference pipeline.",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/inference/baseline.json",
        help="Baseline JSON generated from the training split.",
    )
    parser.add_argument(
        "--split-dir",
        default="data/bccd/valid",
        help="Labeled BCCD split used for calibration.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/calibration",
        help="Directory where calibration artifacts will be written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.40, 0.50],
        help="Confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for matching predictions to ground truth.",
    )
    return parser

"""Calcula las métricas de rendimiento estándar (Precisión, Recall y F1-Score) a 
partir de los conteos de verdaderos positivos, falsos positivos y falsos negativos."""
def compute_detection_metrics(
    *,
    matched: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """Compute precision/recall/F1 from match counts."""
    precision_den = matched + false_positive
    recall_den = matched + false_negative
    precision = matched / precision_den if precision_den else 0.0
    recall = matched / recall_den if recall_den else 0.0
    f1_den = precision + recall
    f1 = (2.0 * precision * recall / f1_den) if f1_den else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

"""Itera sobre los umbrales definidos, evalúa las predicciones contra las etiquetas 
reales y determina el umbral óptimo maximizando el F1-Score, exportando los resultados."""
def main(argv: list[str] | None = None) -> dict[str, object]:
    """Evaluate multiple YOLO confidence thresholds and select the best by F1."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    for threshold in args.thresholds:
        threshold_dir = output_dir / f"thr_{str(threshold).replace('.', 'p')}"
        summary = compare_labels_main(
            [
                "--config",
                args.config,
                "--classifier-run-dir",
                args.classifier_run_dir,
                "--baseline",
                args.baseline,
                "--test-dir",
                args.split_dir,
                "--output-dir",
                str(threshold_dir),
                "--iou-threshold",
                str(args.iou_threshold),
                "--conf-threshold",
                str(threshold),
            ]
        )
        counts = summary["record_counts"]
        matched = int(counts.get("matched", 0))
        false_positive = int(counts.get("false_positive", 0))
        false_negative = int(counts.get("false_negative", 0))
        metrics = compute_detection_metrics(
            matched=matched,
            false_positive=false_positive,
            false_negative=false_negative,
        )
        row = {
            "threshold": threshold,
            "matched": matched,
            "false_positive": false_positive,
            "false_negative": false_negative,
            **metrics,
            "summary_path": str(threshold_dir / "summary.json"),
        }
        rows.append(row)
        if best_row is None or (row["f1"], row["precision"], row["recall"]) > (
            best_row["f1"],
            best_row["precision"],
            best_row["recall"],
        ):
            best_row = row

    csv_path = output_dir / "per_threshold.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "matched",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1",
                "summary_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "split_dir": args.split_dir,
        "iou_threshold": args.iou_threshold,
        "thresholds": rows,
        "best_threshold": best_row["threshold"] if best_row is not None else None,
        "best_metrics": best_row,
        "per_threshold_csv": str(csv_path),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
