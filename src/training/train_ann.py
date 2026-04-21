"""Manual ANN training loop in pure PyTorch (FR-8)."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.ann import ANNFactory, BloodCellANN


@dataclass(frozen=True)
class ANNTrainingConfig:  # pylint: disable=too-few-public-methods
    """Training hyperparameters for the ANN classifier."""

    lr: float
    weight_decay: float


class LossFactory:  # pylint: disable=too-few-public-methods
    """Factory responsible only for constructing loss functions."""

    @staticmethod
    def build() -> nn.Module:
        """Return the multiclass loss used by the ANN."""
        return nn.CrossEntropyLoss()


class OptimizerFactory:  # pylint: disable=too-few-public-methods
    """Factory responsible only for constructing optimizers."""

    @staticmethod
    def build(
        model: nn.Module,
        training_cfg: ANNTrainingConfig,
    ) -> torch.optim.Optimizer:
        """Build the optimizer from training hyperparameters."""
        return torch.optim.Adam(
            model.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )


class ANNTrainer:
    """
    Explicit PyTorch trainer for the ANN classifier.

    FR-8 requires the optimization loop to be coded manually using:
    - forward pass
    - cross-entropy loss
    - loss.backward()
    - optimizer.step()
    """

    def __init__(
        self,
        model: BloodCellANN,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | None = None,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self._device)
        self.criterion = criterion
        self.optimizer = optimizer

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        input_dim: int,
        device: str | None = None,
    ) -> "ANNTrainer":
        """
        Build an ANNTrainer from the project configuration.

        This keeps construction concerns outside __init__ so training
        logic remains decoupled from dependency creation.
        """
        ann_cfg = cfg.get("ann", {})
        training_cfg = ANNTrainingConfig(
            lr=ann_cfg.get("lr", 1e-3),
            weight_decay=ann_cfg.get("weight_decay", 0.0),
        )
        model = ANNFactory.from_project_config(cfg, input_dim=input_dim)
        criterion = LossFactory.build()
        optimizer = OptimizerFactory.build(model, training_cfg)
        return cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """
        Run one explicit training epoch.

        Args:
            dataloader: yields (features, labels).

        Returns:
            Mean loss over the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for features, labels in dataloader:
            features = features.to(self._device)
            labels = labels.to(self._device)

            # FR-8: explicit optimization loop
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return 0.0

        return running_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model without gradient updates.

        Args:
            dataloader: yields (features, labels).

        Returns:
            Dictionary with average loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for features, labels in dataloader:
            features = features.to(self._device)
            labels = labels.to(self._device)

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            running_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        return {
            "loss": running_loss / num_batches,
            "accuracy": correct / total if total > 0 else 0.0,
        }
