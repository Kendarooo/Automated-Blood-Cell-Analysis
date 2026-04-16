"""ANN experiment runner for FR-11 sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.evaluation import MulticlassEvaluator
from src.experiments.feature_pipeline import PreparedFeatureSplits
from src.experiments.run_persistence import build_run_id, persist_run
from src.training.train_ann import ANNTrainer


def run_ann_experiment(
    merged_cfg: dict[str, Any],
    prepared: PreparedFeatureSplits,
) -> dict[str, object]:
    """Run one ANN experiment over precomputed train/validation features."""
    use_wandb = bool(merged_cfg.get("wandb", {}).get("enabled", False))
    started_wandb_run = False
    ann_cfg = merged_cfg.get("ann", {})
    batch_size = int(merged_cfg.get("dataset", {}).get("batch_size", 32))
    epochs = int(ann_cfg.get("epochs", 1))
    device = str(merged_cfg.get("device", "cpu"))
    output_root = merged_cfg.get("outputs", {}).get("runs_dir", "outputs/runs")
    run_id = build_run_id("ann")
    output_dir = torch.path.Path(output_root) / run_id if hasattr(torch, "path") else None
    feature_metadata = {
        "class_names": list(prepared.metadata.class_names),
        "feature_dim": prepared.metadata.feature_dim,
        "truncate_at": prepared.metadata.truncate_at,
        "projection_dim": prepared.metadata.projection_dim,
        "normalization_enabled": prepared.metadata.normalization_enabled,
        "dimensionality_strategy": prepared.metadata.dimensionality_strategy,
    }

    if use_wandb and wandb.run is None:
        wandb_cfg = merged_cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config={
                "ann": ann_cfg,
                "dataset": {"batch_size": batch_size},
                "device": device,
                "feature_metadata": feature_metadata,
            },
        )
        started_wandb_run = True

    train_loader = _build_dataloader(
        features=prepared.train.features,
        labels=prepared.train.labels,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _build_dataloader(
        features=prepared.val.features,
        labels=prepared.val.labels,
        batch_size=batch_size,
        shuffle=False,
    )

    trainer = ANNTrainer.from_config(
        merged_cfg,
        input_dim=prepared.metadata.feature_dim,
        device=device,
    )

    train_loss = 0.0
    for _ in range(epochs):
        train_loss = trainer.train_one_epoch(train_loader)

    val_summary = trainer.evaluate(val_loader)
    predictions = _predict_labels(
        trainer=trainer,
        features=prepared.val.features,
        device=device,
    )
    evaluator = MulticlassEvaluator(prepared.metadata.class_names)
    metrics = evaluator.evaluate(prepared.val.labels, predictions)
    metrics_payload = {
        "train_loss": float(train_loss),
        "val_loss": float(val_summary["loss"]),
        "val_accuracy": float(metrics.accuracy),
        "val_macro_f1": float(metrics.macro_f1),
        "val_precision_per_class": metrics.precision_per_class,
        "val_recall_per_class": metrics.recall_per_class,
        "val_confusion_matrix": metrics.confusion_matrix.tolist(),
    }
    result = {
        **metrics_payload,
        "feature_metadata": feature_metadata,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    if use_wandb:
        log_payload: dict[str, object] = {
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
            "val_macro_f1": result["val_macro_f1"],
            "val_confusion_matrix": result["val_confusion_matrix"],
        }
        for class_name, value in metrics.precision_per_class.items():
            log_payload[f"val_precision_{class_name}"] = float(value)
        for class_name, value in metrics.recall_per_class.items():
            log_payload[f"val_recall_{class_name}"] = float(value)

        wandb.log(log_payload)
        if started_wandb_run:
            wandb.finish()

    from pathlib import Path

    resolved_output_dir = Path(output_root) / run_id
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolved_output_dir / "model.pt"
    torch.save(trainer.model.state_dict(), model_path)

    persistence_info = persist_run(
        run_id=run_id,
        classifier_name="ann",
        output_root=output_root,
        effective_config=merged_cfg,
        metrics=metrics_payload,
        feature_metadata=feature_metadata,
        model_path=model_path,
        seed=merged_cfg.get("seed"),
        wandb_metadata={
            "enabled": use_wandb,
            "project": merged_cfg.get("wandb", {}).get("project"),
            "entity": merged_cfg.get("wandb", {}).get("entity"),
            "run_name": merged_cfg.get("wandb", {}).get("run_name"),
        },
    )
    result.update(persistence_info)

    return result


def _build_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_labels(
    trainer: ANNTrainer,
    features: np.ndarray,
    device: str,
) -> np.ndarray:
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    logits = trainer.model(feature_tensor)
    return logits.argmax(dim=1).detach().cpu().numpy()
