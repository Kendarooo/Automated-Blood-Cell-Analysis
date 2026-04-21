"""CLI entrypoint for YOLO fine-tuning on the BCCD dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed
from src.utils.wandb_logger import WandBLogger

os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(Path(".ultralytics").resolve()))

from src.detection.yolo_train import YOLOTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for YOLO fine-tuning."""
    parser = argparse.ArgumentParser(description="Train YOLO on BCCD.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the base project config file.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Run training without logging to W&B.",
    )
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--imgsz", type=int, help="Override training image size.")
    parser.add_argument("--batch", type=int, help="Override training batch size.")
    parser.add_argument("--lr0", type=float, help="Override base learning rate.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute one YOLO fine-tuning run and print a structured summary."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.disable_wandb:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["enabled"] = False

    detection_cfg = cfg.setdefault("detection", {})
    for key in ("epochs", "imgsz", "batch", "lr0"):
        value = getattr(args, key)
        if value is not None:
            detection_cfg[key] = value

    set_seed(int(cfg.get("seed", 42)))
    ensure_dir(detection_cfg.get("output_dir", "outputs"))

    with WandBLogger(cfg) as logger:
        trainer = YOLOTrainer(cfg, logger=logger)
        best_weights = trainer.train()
        wandb_run_id = logger.run_id
        wandb_run_url = logger.run_url

    result = {
        "best_weights": str(best_weights),
        "best_weights_exists": Path(best_weights).exists(),
        "run_summary_path": str(best_weights.parent.parent / "run_summary.json"),
        "effective_config_path": str(best_weights.parent.parent / "effective_config.json"),
        "data_yaml": detection_cfg.get("data_yaml"),
        "weights": detection_cfg.get("weights"),
        "epochs": detection_cfg.get("epochs"),
        "imgsz": detection_cfg.get("imgsz"),
        "batch": detection_cfg.get("batch"),
        "lr0": detection_cfg.get("lr0"),
        "wandb_enabled": bool(cfg.get("wandb", {}).get("enabled", True)),
        "wandb_run_id": wandb_run_id,
        "wandb_run_url": wandb_run_url,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
