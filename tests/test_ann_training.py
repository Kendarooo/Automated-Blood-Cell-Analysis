"""Unit tests for explicit ANN training loop (FR-8, NFR-6d)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.train_ann import ANNTrainer


def test_ann_training_produces_nonzero_gradients() -> None:
    """
    NFR-6d / FR-8:
    after at least one training iteration, the ANN should produce non-zero
    gradients through the explicit PyTorch optimization loop.
    """
    cfg = {
        "ann": {
            "hidden_dims": [16, 8],
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "num_classes": 3,
        }
    }

    features = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.1, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.6],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = ANNTrainer.from_config(cfg, input_dim=4, device="cpu")
    loss = trainer.train_one_epoch(dataloader)

    assert loss > 0.0

    gradients = [
        param.grad
        for param in trainer.model.parameters()
        if param.grad is not None
    ]

    assert gradients, "Expected at least one parameter to have gradients."
    assert any(torch.any(grad != 0) for grad in gradients), (
        "Expected at least one non-zero gradient after one training epoch."
    )
