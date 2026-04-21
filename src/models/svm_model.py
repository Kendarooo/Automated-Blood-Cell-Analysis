"""SVM model construction for multiclass blood-cell classification (FR-9)."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class SVMConfig:
    """Immutable SVM hyperparameter configuration."""

    kernel: str
    c_value: float
    gamma: str | float


class SVMFactory:
    """
    Factory responsible only for building multiclass SVM classifiers.

    Supports the kernels required by FR-9:
    - linear
    - rbf
    """

    @staticmethod
    def from_project_config(cfg: dict) -> OneVsRestClassifier:
        """Build an SVM from the project configuration dictionary."""
        svm_cfg = cfg["svm"]
        model_cfg = SVMConfig(
            kernel=svm_cfg.get("kernel", "rbf"),
            c_value=svm_cfg.get("C", 1.0),
            gamma=svm_cfg.get("gamma", "scale"),
        )
        return SVMFactory.build(model_cfg)

    @staticmethod
    def build(model_cfg: SVMConfig) -> OneVsRestClassifier:
        """
        Build a multiclass One-vs-Rest SVM.

        Args:
            model_cfg: Immutable SVM hyperparameter configuration.

        Returns:
            Multiclass SVM classifier ready for fit/predict.
        """
        if model_cfg.kernel not in {"linear", "rbf"}:
            raise ValueError(
                f"Unsupported kernel '{model_cfg.kernel}'. "
                "Expected 'linear' or 'rbf'."
            )

        estimator = SVC(
            kernel=model_cfg.kernel,
            C=model_cfg.c_value,
            gamma=model_cfg.gamma,
        )
        return OneVsRestClassifier(estimator)
