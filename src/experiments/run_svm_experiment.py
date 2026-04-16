"""SVM experiment runner for FR-11 sweeps."""

from __future__ import annotations

from typing import Any

import wandb
import joblib

from src.experiments.evaluation import MulticlassEvaluator
from src.experiments.feature_pipeline import PreparedFeatureSplits
from src.experiments.run_persistence import build_run_id, persist_run
from src.models.svm_model import SVMConfig
from src.training.train_svm import DatasetSplit, SVMTrainer


def run_svm_experiment(
    merged_cfg: dict[str, Any],
    prepared: PreparedFeatureSplits,
) -> dict[str, object]:
    """Run one SVM experiment over precomputed train/validation features."""
    use_wandb = bool(merged_cfg.get("wandb", {}).get("enabled", False))
    started_wandb_run = False
    svm_cfg = merged_cfg.get("svm", {})
    has_point_config = all(
        key in svm_cfg for key in ("kernel", "C", "gamma")
    )
    output_root = merged_cfg.get("outputs", {}).get("runs_dir", "outputs/runs")
    run_id = build_run_id("svm")
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
            settings=wandb.Settings(x_disable_viewer=True, silent=True),
            config={
                "svm": svm_cfg,
                "feature_metadata": feature_metadata,
            },
        )
        started_wandb_run = True

    train_split = DatasetSplit(
        features=prepared.train.features,
        labels=prepared.train.labels,
    )
    val_split = DatasetSplit(
        features=prepared.val.features,
        labels=prepared.val.labels,
    )

    trainer = SVMTrainer()
    if has_point_config:
        trained_runs = [
            trainer._validator.train_and_evaluate(  # noqa: SLF001
                train_split=train_split,
                val_split=val_split,
                config=SVMConfig(
                    kernel=str(svm_cfg["kernel"]),
                    c_value=float(svm_cfg["C"]),
                    gamma=svm_cfg["gamma"],
                ),
            )
        ]
        best_run = trained_runs[0]
    else:
        c_values = list(svm_cfg.get("c_values", [svm_cfg.get("C", 1.0)]))
        gamma_values = list(svm_cfg.get("gamma_values", [svm_cfg.get("gamma", "scale")]))
        trained_runs = trainer.compare_kernels_with_models(
            train_split=train_split,
            val_split=val_split,
            c_values=c_values,
            gamma_values=gamma_values,
        )
        best_run = trainer.select_best_trained(trained_runs)
    predictions = best_run.classifier.predict(val_split.features)

    evaluator = MulticlassEvaluator(prepared.metadata.class_names)
    metrics = evaluator.evaluate(prepared.val.labels, predictions)
    metrics_payload = {
        "val_accuracy": float(metrics.accuracy),
        "val_macro_f1": float(metrics.macro_f1),
        "val_precision_per_class": metrics.precision_per_class,
        "val_recall_per_class": metrics.recall_per_class,
        "val_confusion_matrix": metrics.confusion_matrix.tolist(),
    }
    result = {
        **metrics_payload,
        "selected_kernel": best_run.result.kernel,
        "selected_c_value": best_run.result.c_value,
        "selected_gamma": best_run.result.gamma,
        "num_evaluated_configs": len(trained_runs),
        "feature_metadata": feature_metadata,
    }

    if use_wandb:
        log_payload: dict[str, object] = {
            "val_accuracy": result["val_accuracy"],
            "val_macro_f1": result["val_macro_f1"],
            "val_confusion_matrix": result["val_confusion_matrix"],
            "selected_kernel": result["selected_kernel"],
            "selected_c_value": result["selected_c_value"],
            "selected_gamma": result["selected_gamma"],
            "num_evaluated_configs": result["num_evaluated_configs"],
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
    model_path = resolved_output_dir / "model.joblib"
    joblib.dump(best_run.classifier, model_path)

    persistence_info = persist_run(
        run_id=run_id,
        classifier_name="svm",
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
