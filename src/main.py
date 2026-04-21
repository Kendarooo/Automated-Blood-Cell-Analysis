"""Top-level pipeline orchestrator for training and inference stages."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import wandb

from src.pipeline_stages import (
    ANNTrainingStage,
    BaselineStage,
    DetectorTrainingStage,
    FeatureExtractionStage,
    InferenceStage,
    SVMTrainingStage,
)
from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed

"""Configura los argumentos de la terminal para definir la etapa del pipeline, 
rutas de configuración y ejecución de sweeps."""
def build_parser() -> argparse.ArgumentParser:
    """Build the project-level CLI parser."""
    parser = argparse.ArgumentParser(description="Run the BCCD pipeline stages.")
    parser.add_argument(
        "--stage",
        choices=(
            "train_detector", "extract_features", "train_ann", "train_svm",
            "build_baseline", "infer_image", "all",
        ),
        required=True,
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--image", help="Image path for infer_image stage.")
    parser.add_argument("--classifier-run-dir", help="Persisted classifier run directory.")
    parser.add_argument("--baseline", default="outputs/inference/baseline.json")
    return parser

"""Inicializa el entorno de ejecución (semilla, directorios, W&B) y delega el 
flujo a la etapa solicitada, retornando los resultados en JSON."""
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

"""Instancia y ejecuta la etapa específica seleccionada, gestionando el paso de 
datos si se ejecutan múltiples etapas en secuencia ('all')."""
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
        return feature_stage.run(cfg)["artifacts"]

    if args.stage == "train_ann":
        prepared = feature_stage.run(cfg)["prepared"]
        return ann_stage.run(cfg, prepared)

    if args.stage == "train_svm":
        prepared = feature_stage.run(cfg)["prepared"]
        return svm_stage.run(cfg, prepared)

    if args.stage == "build_baseline":
        return baseline_stage.run(cfg, output_path=args.baseline)

    if args.stage == "infer_image":
        return inference_stage.run(
            cfg,
            image_path=_require_argument(args.image, "--image is required for infer_image."),
            classifier_run_dir=_require_argument(
                args.classifier_run_dir, "--classifier-run-dir is required for infer_image."
            ),
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

"""Inicia el seguimiento de experimentos en W&B y sobrescribe la configuración
base con los parámetros del sweep actual si está activo."""
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

"""Funciones utilitarias para inyectar valores de claves aplanadas (ej. 'param.subparam') 
dentro de la jerarquía del diccionario de configuración."""
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

"""Valida que los argumentos obligatorios de la CLI estén presentes para evitar errores de ejecución."""
def _require_argument(value: str | None, error_message: str) -> str:
    """Return an argument value or raise a clear CLI error."""
    if value is None:
        raise ValueError(error_message)
    return value


if __name__ == "__main__":
    main()
