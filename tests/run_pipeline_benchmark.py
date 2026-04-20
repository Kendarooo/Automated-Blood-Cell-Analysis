"""Standalone benchmark utility for pipeline timing measurements."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import ValTransforms
from src.detection.yolo_infer import YOLOInferencer
from src.features.extractor import ResNet18Extractor
from src.features.normalize import FeatureNormalizer
from src.inference.artifact_loader import ArtifactLoader, ArtifactPaths
from src.inference.pipeline import InferencePipeline
from src.inference.run_loader import load_run
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline stages over the test split.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project configuration file.",
    )
    parser.add_argument(
        "--ann-run-dir",
        default="outputs/runs/ann_0034986f",
        help="Persisted ANN run directory used for inference timing.",
    )
    parser.add_argument(
        "--svm-run-dir",
        default="outputs/runs/svm_f67e2c6c",
        help="Persisted SVM run directory used for inference timing.",
    )
    parser.add_argument(
        "--baseline",
        default="outputs/inference/baseline.json",
        help="Baseline JSON artifact for inference.",
    )
    parser.add_argument(
        "--images-dir",
        default="data/bccd/test/images",
        help="Directory containing benchmark images.",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark/pipeline_benchmark_summary.json",
        help="Where to save the benchmark summary JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the benchmark and print/save a structured summary."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    images = sorted(Path(args.images_dir).glob("*.jpg"))
    if not images:
        raise ValueError(f"No .jpg images found in benchmark directory '{args.images_dir}'.")

    ann_run_dir = Path(args.ann_run_dir)
    svm_run_dir = Path(args.svm_run_dir)
    baseline_path = Path(args.baseline)

    detector = YOLOInferencer(cfg)
    transform = ValTransforms(cfg)
    extractor = ResNet18Extractor(cfg).eval()
    artifact_loader = ArtifactLoader()
    ann_artifacts = artifact_loader.load(
        ArtifactPaths(
            detector=Path(cfg["detection"]["best_weights"]),
            extractor_config=ann_run_dir / "effective_config.json",
            classifier=None,
            normalizer=ann_run_dir / "normalizer.npz",
            baseline=baseline_path,
            classifier_run_dir=ann_run_dir,
        ),
        cfg,
    )
    svm_loaded_run = load_run(svm_run_dir)
    svm_normalizer = FeatureNormalizer.load(str(svm_run_dir / "normalizer.npz"))
    pipeline = InferencePipeline(
        cfg,
        ArtifactPaths(
            detector=Path(cfg["detection"]["best_weights"]),
            extractor_config=ann_run_dir / "effective_config.json",
            classifier=None,
            normalizer=ann_run_dir / "normalizer.npz",
            baseline=baseline_path,
            classifier_run_dir=ann_run_dir,
        ),
    )

    _warm_up(images[0], detector, transform, extractor, ann_artifacts, svm_loaded_run, svm_normalizer, pipeline)
    summary = _run_measurements(
        images=images,
        detector=detector,
        transform=transform,
        extractor=extractor,
        ann_artifacts=ann_artifacts,
        svm_loaded_run=svm_loaded_run,
        svm_normalizer=svm_normalizer,
        pipeline=pipeline,
    )

    summary.update(
        {
            "config_path": str(Path(args.config)),
            "ann_run_dir": str(ann_run_dir),
            "svm_run_dir": str(svm_run_dir),
            "baseline_path": str(baseline_path),
            "images_dir": str(Path(args.images_dir)),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(output_path)}, indent=2))
    return summary


def _warm_up(
    image_path: Path,
    detector: YOLOInferencer,
    transform: ValTransforms,
    extractor: ResNet18Extractor,
    ann_artifacts: Any,
    svm_loaded_run: Any,
    svm_normalizer: FeatureNormalizer,
    pipeline: InferencePipeline,
) -> None:
    """Run one warm-up pass so initialization time is not benchmarked."""
    detection = detector.predict(image_path)
    if detection.crops:
        batch = torch.stack([transform(crop) for crop in detection.crops])
        with torch.no_grad():
            features = extractor(batch).detach().cpu().numpy()
        ann_features = ann_artifacts.normalizer.transform(features)
        with torch.no_grad():
            ann_artifacts.classifier(torch.tensor(ann_features, dtype=torch.float32))
        svm_loaded_run.model.predict(svm_normalizer.transform(features))
    pipeline.predict(image_path)


def _run_measurements(
    *,
    images: list[Path],
    detector: YOLOInferencer,
    transform: ValTransforms,
    extractor: ResNet18Extractor,
    ann_artifacts: Any,
    svm_loaded_run: Any,
    svm_normalizer: FeatureNormalizer,
    pipeline: InferencePipeline,
) -> dict[str, Any]:
    """Measure each pipeline stage over the provided image set."""
    detection_times: list[float] = []
    feature_times: list[float] = []
    ann_times: list[float] = []
    svm_times: list[float] = []
    full_pipeline_times: list[float] = []

    for image in images:
        start = time.perf_counter()
        detection = detector.predict(image)
        detection_times.append(time.perf_counter() - start)

        if detection.crops:
            start = time.perf_counter()
            image_batch = torch.stack([transform(crop) for crop in detection.crops])
            with torch.no_grad():
                raw_features = extractor(image_batch).detach().cpu().numpy()
            feature_times.append(time.perf_counter() - start)

            ann_features = ann_artifacts.normalizer.transform(raw_features)
            start = time.perf_counter()
            with torch.no_grad():
                ann_artifacts.classifier(torch.tensor(ann_features, dtype=torch.float32))
            ann_times.append(time.perf_counter() - start)

            svm_features = svm_normalizer.transform(raw_features)
            start = time.perf_counter()
            svm_loaded_run.model.predict(svm_features)
            svm_times.append(time.perf_counter() - start)
        else:
            feature_times.append(0.0)
            ann_times.append(0.0)
            svm_times.append(0.0)

        start = time.perf_counter()
        pipeline.predict(image)
        full_pipeline_times.append(time.perf_counter() - start)

    return {
        "num_images": len(images),
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "disk_policy": {
            "detection": "includes image read from disk",
            "feature_extraction": "excludes disk load; uses crops already in memory",
            "classification_ann": "excludes disk load; uses normalized features already in memory",
            "classification_svm": "excludes disk load; uses normalized features already in memory",
            "full_pipeline": "includes image read from disk, excludes model/artifact load",
        },
        "timings_seconds": {
            "detection_mean": statistics.mean(detection_times),
            "detection_std": statistics.pstdev(detection_times),
            "feature_mean": statistics.mean(feature_times),
            "feature_std": statistics.pstdev(feature_times),
            "ann_mean": statistics.mean(ann_times),
            "ann_std": statistics.pstdev(ann_times),
            "svm_mean": statistics.mean(svm_times),
            "svm_std": statistics.pstdev(svm_times),
            "pipeline_mean": statistics.mean(full_pipeline_times),
            "pipeline_std": statistics.pstdev(full_pipeline_times),
        },
    }


if __name__ == "__main__":
    main()
