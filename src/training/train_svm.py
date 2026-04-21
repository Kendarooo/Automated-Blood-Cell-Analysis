"""SVM training and validation comparison for FR-9."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

from src.models.svm_model import SVMConfig, SVMFactory


@dataclass(frozen=True)
class SVMEvaluationResult:  # pylint: disable=too-few-public-methods
    """Evaluation summary for one SVM configuration."""

    kernel: str
    c_value: float
    gamma: str | float
    accuracy: float


@dataclass(frozen=True)
class TrainedSVMRun:  # pylint: disable=too-few-public-methods
    """Evaluation result bundled with the trained classifier instance."""

    result: SVMEvaluationResult
    classifier: OneVsRestClassifier


@dataclass(frozen=True)
class DatasetSplit:
    """Feature matrix and labels for one dataset split."""

    features: np.ndarray
    labels: np.ndarray


class SVMValidator:
    """
    Responsible only for fitting on train and evaluating on validation.

    This class does not build search spaces; it only evaluates a given
    configuration against the validation split.
    """

    def evaluate(
        self,
        train_split: DatasetSplit,
        val_split: DatasetSplit,
        config: SVMConfig,
    ) -> SVMEvaluationResult:
        """Train one SVM configuration on train and score it on validation."""
        classifier = SVMFactory.build(config)
        classifier.fit(train_split.features, train_split.labels)
        predictions = classifier.predict(val_split.features)
        accuracy = accuracy_score(val_split.labels, predictions)

        return SVMEvaluationResult(
            kernel=config.kernel,
            c_value=config.c_value,
            gamma=config.gamma,
            accuracy=float(accuracy),
        )

    def train_and_evaluate(
        self,
        train_split: DatasetSplit,
        val_split: DatasetSplit,
        config: SVMConfig,
    ) -> TrainedSVMRun:
        """Train one SVM configuration and keep the fitted classifier."""
        classifier = SVMFactory.build(config)
        classifier.fit(train_split.features, train_split.labels)
        predictions = classifier.predict(val_split.features)
        accuracy = accuracy_score(val_split.labels, predictions)

        return TrainedSVMRun(
            result=SVMEvaluationResult(
                kernel=config.kernel,
                c_value=config.c_value,
                gamma=config.gamma,
                accuracy=float(accuracy),
            ),
            classifier=classifier,
        )


class SVMTrainer:
    """
    Coordinates comparison of linear vs. RBF SVMs on the validation set.

    The validation set is used only for model selection, never for fitting.
    """

    def __init__(self, validator: SVMValidator | None = None) -> None:
        self._validator = validator or SVMValidator()

    def compare_kernels(
        self,
        train_split: DatasetSplit,
        val_split: DatasetSplit,
        c_values: list[float],
        gamma_values: list[str | float],
    ) -> list[SVMEvaluationResult]:
        """
        Compare linear and RBF kernels using validation accuracy.

        Returns:
            List of evaluation results, one per explored configuration.
        """
        results: list[SVMEvaluationResult] = []

        for c_value in c_values:
            results.append(
                self._validator.evaluate(
                    train_split,
                    val_split,
                    SVMConfig(kernel="linear", c_value=c_value, gamma="scale"),
                )
            )

            for gamma in gamma_values:
                results.append(
                    self._validator.evaluate(
                        train_split,
                        val_split,
                        SVMConfig(kernel="rbf", c_value=c_value, gamma=gamma),
                    )
                )

        return results

    def compare_kernels_with_models(
        self,
        train_split: DatasetSplit,
        val_split: DatasetSplit,
        c_values: list[float],
        gamma_values: list[str | float],
    ) -> list[TrainedSVMRun]:
        """
        Compare linear and RBF kernels while retaining fitted classifiers.

        Returns:
            List of trained runs, one per explored configuration.
        """
        runs: list[TrainedSVMRun] = []

        for c_value in c_values:
            runs.append(
                self._validator.train_and_evaluate(
                    train_split,
                    val_split,
                    SVMConfig(kernel="linear", c_value=c_value, gamma="scale"),
                )
            )

            for gamma in gamma_values:
                runs.append(
                    self._validator.train_and_evaluate(
                        train_split,
                        val_split,
                        SVMConfig(kernel="rbf", c_value=c_value, gamma=gamma),
                    )
                )

        return runs

    def select_best(
        self,
        results: list[SVMEvaluationResult],
    ) -> SVMEvaluationResult:
        """Return the best validation result."""
        if not results:
            raise ValueError("No SVM evaluation results were provided.")

        return max(results, key=lambda result: result.accuracy)

    def select_best_trained(
        self,
        runs: list[TrainedSVMRun],
    ) -> TrainedSVMRun:
        """Return the best trained run."""
        if not runs:
            raise ValueError("No trained SVM runs were provided.")

        return max(runs, key=lambda run: run.result.accuracy)
