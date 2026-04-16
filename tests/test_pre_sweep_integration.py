"""Pre-sweep integration test for the FR-11 experiment pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.evaluation import MulticlassEvaluator
from src.experiments.feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
)
from src.models.svm_model import SVMConfig, SVMFactory
from src.training.train_ann import ANNTrainer
from src.training.train_svm import DatasetSplit


class RecordingNormalizer:
    """Normalizer double that records train-only fitting behavior."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.transform_calls = 0
        self.fit_inputs: list[np.ndarray] = []
        self.transform_inputs: list[np.ndarray] = []

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit_calls += 1
        self.fit_inputs.append(features.copy())
        return features

    def transform(self, features: np.ndarray) -> np.ndarray:
        self.transform_calls += 1
        self.transform_inputs.append(features.copy())
        return features


class StubSplitLoader:
    """Split loader double that exposes only train and validation."""

    def __init__(self) -> None:
        self.called_split_dirs: list[Path] = []

    def load_split(
        self,
        split_dir: Path,
        class_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, np.ndarray]:
        del class_names
        self.called_split_dirs.append(split_dir)

        if split_dir.name == "train":
            images = torch.tensor(
                [[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]],
                dtype=torch.float32,
            ).repeat(1, 3, 224, 224)
            labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
            return images, labels

        if split_dir.name == "val":
            images = torch.tensor(
                [[[[0.2]]], [[[2.2]]], [[[4.8]]]],
                dtype=torch.float32,
            ).repeat(1, 3, 224, 224)
            labels = np.array([0, 1, 2], dtype=np.int64)
            return images, labels

        raise AssertionError(f"Unexpected split requested: {split_dir}")


class StubExtractor:
    """Extractor double faithful to FR-5 dimensions and sensitive to image values."""

    def __init__(self, truncate_at: str) -> None:
        dims = {
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }
        self.output_dim = dims[truncate_at]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        base_signal = inputs[:, 0, 0, 0].unsqueeze(1)
        row = torch.arange(self.output_dim, dtype=torch.float32).unsqueeze(0)
        return base_signal + row.repeat(batch_size, 1) * 0.01


def test_pre_sweep_pipeline_is_valid() -> None:
    """
    Validate the minimum experiment path before launching W&B sweeps.

    This test enforces that:
    - features are prepared only from train/val
    - normalization fits only on train
    - ANN and SVM consume the same feature space
    - shared evaluation produces finite metrics
    """
    config = FeaturePipelineConfig(
        train_dir=Path("train"),
        val_dir=Path("val"),
        truncate_at="layer3",
        projection_dim=None,
        normalization_enabled=True,
        dimensionality_strategy="none",
        class_names=("Platelets", "RBC", "WBC"),
    )
    split_loader = StubSplitLoader()
    normalizer = RecordingNormalizer()
    pipeline = FeaturePipeline(
        config=config,
        split_loader=split_loader,
        extractor=StubExtractor("layer3"),
        normalizer=normalizer,
    )

    prepared = pipeline.prepare_train_val_features()

    assert prepared.train.features.shape == (6, 256)
    assert prepared.val.features.shape == (3, 256)
    assert prepared.metadata.feature_dim == 256
    assert normalizer.fit_calls == 1
    assert normalizer.transform_calls == 1
    assert not np.array_equal(
        normalizer.fit_inputs[0],
        normalizer.transform_inputs[0],
    )
    assert [path.name for path in split_loader.called_split_dirs] == ["train", "val"]

    ann_cfg = {
        "ann": {
            "hidden_dims": [12],
            "dropout": 0.0,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "num_classes": 3,
        }
    }
    train_dataset = TensorDataset(
        torch.tensor(prepared.train.features, dtype=torch.float32),
        torch.tensor(prepared.train.labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(prepared.val.features, dtype=torch.float32),
        torch.tensor(prepared.val.labels, dtype=torch.long),
    )
    ann_trainer = ANNTrainer.from_config(ann_cfg, input_dim=prepared.metadata.feature_dim, device="cpu")
    ann_train_loss = ann_trainer.train_one_epoch(DataLoader(train_dataset, batch_size=2, shuffle=False))
    ann_summary = ann_trainer.evaluate(DataLoader(val_dataset, batch_size=3, shuffle=False))
    ann_logits = ann_trainer.model(torch.tensor(prepared.val.features, dtype=torch.float32))
    ann_predictions = ann_logits.argmax(dim=1).detach().cpu().numpy()

    svm_classifier = SVMFactory.build(
        SVMConfig(kernel="linear", c_value=1.0, gamma="scale")
    )
    train_split = DatasetSplit(features=prepared.train.features, labels=prepared.train.labels)
    val_split = DatasetSplit(features=prepared.val.features, labels=prepared.val.labels)
    svm_classifier.fit(train_split.features, train_split.labels)
    svm_predictions = svm_classifier.predict(val_split.features)

    evaluator = MulticlassEvaluator(prepared.metadata.class_names)
    ann_eval = evaluator.evaluate(prepared.val.labels, ann_predictions)
    svm_eval = evaluator.evaluate(prepared.val.labels, svm_predictions)

    assert ann_train_loss >= 0.0
    assert 0.0 <= ann_summary["accuracy"] <= 1.0
    assert prepared.train.features.shape[1] == prepared.val.features.shape[1]
    assert prepared.metadata.feature_dim == train_split.features.shape[1]
    assert train_split.features.shape[1] == val_split.features.shape[1]
    assert not np.isnan(ann_eval.macro_f1)
    assert not np.isnan(svm_eval.macro_f1)
    assert ann_eval.confusion_matrix.shape == (3, 3)
    assert svm_eval.confusion_matrix.shape == (3, 3)
