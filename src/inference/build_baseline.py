"""CLI to estimate and persist a train-derived baseline distribution."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.baseline import BaselineEstimator, BaselineRepository
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for baseline estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate a baseline distribution from BCCD train labels.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project configuration file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference/baseline.json",
        help="Where to save the baseline JSON artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Estimate and save the baseline distribution from train labels only."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    class_names = list(dataset_cfg.get("class_names", ["Platelets", "RBC", "WBC"]))
    train_dir = Path(dataset_cfg.get("train_dir", "data/bccd/train"))

    train_counts = _load_train_counts(train_dir, class_names)
    baseline = BaselineEstimator(class_names).fit_from_train_counts(train_counts, cfg)
    BaselineRepository.save(baseline, args.output)

    result = {
        "output_path": args.output,
        "num_train_images": len(train_counts),
        "class_order": baseline.class_order,
        "proportions": baseline.proportions,
        "created_from_split": baseline.created_from_split,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_train_counts(
    train_dir: Path,
    class_names: list[str],
) -> list[dict[str, int]]:
    """Load per-image class counts from YOLO-format train labels."""
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


if __name__ == "__main__":
    main()
