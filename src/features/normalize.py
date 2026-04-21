"""Feature vector normalizer for the BCCD classification pipeline (FR-6, NFR-4).

Critical design constraint: the normalizer must be fit EXCLUSIVELY on
training data. Applying fit() on validation or test data would leak
statistical information and invalidate generalization metrics (NFR-4).
"""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

import numpy as np


class FeatureNormalizer:
    """
    Single Responsibility: standardize feature vectors to zero mean
    and unit variance.

    Usage pattern (NFR-4 compliant):
        normalizer = FeatureNormalizer()
        X_train_norm = normalizer.fit_transform(X_train)  # fit only here
        X_val_norm   = normalizer.transform(X_val)        # never fit here
        X_test_norm  = normalizer.transform(X_test)       # never fit here
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """
        Compute mean and std from training data only (FR-6).

        Args:
            X: Feature matrix of shape (N, K) — training set only.

        Returns:
            self, to allow method chaining.
        """
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)

        # Avoid division by zero for constant features
        self._std[self._std == 0] = 1.0
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization using previously fitted parameters.

        Args:
            X: Feature matrix of shape (N, K).

        Returns:
            Standardized feature matrix of shape (N, K).

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_fitted()
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on X and return its standardized version.
        Convenience method — should ONLY be called with training data.

        Args:
            X: Feature matrix of shape (N, K) — training set only.

        Returns:
            Standardized feature matrix of shape (N, K).
        """
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """
        Persist fitted parameters to disk (FR-13).

        Args:
            path: File path without extension. Saves as .npz.
        """
        self._check_fitted()
        np.savez(path, mean=self._mean, std=self._std)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """
        Load fitted parameters from disk.

        Args:
            path: Path to the .npz file saved by save().

        Returns:
            Fitted FeatureNormalizer instance ready to call transform().
        """
        data = np.load(path)
        normalizer = cls()
        normalizer._mean = data["mean"]  # pylint: disable=protected-access
        normalizer._std = data["std"]    # pylint: disable=protected-access
        normalizer._is_fitted = True     # pylint: disable=protected-access
        return normalizer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if transform() is called before fit()."""
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureNormalizer has not been fitted yet. "
                "Call fit() or fit_transform() with training data first."
            )
