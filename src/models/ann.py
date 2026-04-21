"""Artificial Neural Network for multiclass blood-cell classification (FR-8)."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

"""Estructura inmutable que define la topología y los hiperparámetros de la red 
(dimensión de entrada, capas ocultas, dropout y número de clases)."""
@dataclass(frozen=True)
class ANNArchitectureConfig:  # pylint: disable=too-few-public-methods
    """Immutable ANN architecture configuration."""

    input_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    num_classes: int

"""Clase fábrica (Factory) encargada exclusivamente de instanciar el modelo de 
PyTorch mapeando los valores desde el diccionario de configuración del proyecto."""
class ANNFactory:  # pylint: disable=too-few-public-methods
    """Factory responsible only for building ANN models from configuration."""

    @staticmethod
    def from_project_config(cfg: dict, input_dim: int) -> "BloodCellANN":
        """Create a model from the project config dictionary."""
        ann_cfg = cfg["ann"]
        architecture = ANNArchitectureConfig(
            input_dim=input_dim,
            hidden_dims=tuple(ann_cfg.get("hidden_dims", [128, 64])),
            dropout=ann_cfg.get("dropout", 0.3),
            num_classes=ann_cfg.get("num_classes", 3),
        )
        return BloodCellANN(architecture)

"""Implementación de la red neuronal multiclase, que define la secuencia de capas 
lineales, activaciones (ReLU), regularización (Dropout) y el cálculo de la 
propagación hacia adelante (forward pass)."""
class BloodCellANN(nn.Module):
    """
    Multiclass ANN classifier implemented in pure PyTorch.

    This module only defines the network architecture.
    The explicit optimization loop required by FR-8 lives in the
    training module, not inside this class.
    """

    def __init__(self, architecture: ANNArchitectureConfig) -> None:
        """
        Args:
            architecture: Immutable description of the ANN topology.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = architecture.input_dim

        for hidden_dim in architecture.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(architecture.dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, architecture.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for each class.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.network(x)
