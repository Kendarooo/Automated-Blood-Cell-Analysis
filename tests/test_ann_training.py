"""Unit tests for explicit ANN training loop (FR-8, NFR-6d)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ann import ANNArchitectureConfig, BloodCellANN
from src.training.train_ann import ANNTrainer


def test_fr8_one_step_all_params_have_nonzero_gradients() -> None:
    """
    FR-8: after ONE explicit optimisation step the gradient of EVERY
    trainable parameter must be non-zero.

    Dropout is set to 0.0 so no units are masked; all paths are active
    and every weight must receive signal from backward().
    """
    K = 32   # arbitrary feature dimensionality
    architecture = ANNArchitectureConfig(
        input_dim=K,
        hidden_dims=(64, 32),
        dropout=0.0,       # no masking → all gradients populated
        num_classes=3,
    )
    model = BloodCellANN(architecture)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic batch: 12 samples covering all 3 classes
    X = torch.randn(12, K)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)

    # --- Explicit optimisation step (FR-8 requirement) ---
    model.train()
    optimizer.zero_grad()
    predictions = model(X)                   # a) forward pass
    loss = criterion(predictions, y)         # b) cross-entropy loss
    loss.backward()                          # c) backprop
    optimizer.step()                         # d) weight update

    assert loss.item() > 0.0, "Loss must be positive."

    for name, param in model.named_parameters():
        assert param.grad is not None, (
            f"Parameter '{name}' has no gradient after backward()."
        )
        assert torch.any(param.grad != 0), (
            f"Parameter '{name}' has an all-zero gradient after backward()."
        )


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
