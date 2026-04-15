"""FR-6 tests for feature normalization using train-only statistics."""

import numpy as np

from src.features.normalize import FeatureNormalizer


def test_fr6_normalizer_fits_on_train_and_transforms_test_without_refit() -> None:
    """
    FR-6:
    fit statistics on train only, reuse them for test, and do not refit on test.
    """
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
    ])
    X_test = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 22.0, 33.0],
    ])

    normalizer = FeatureNormalizer()

    X_train_norm = normalizer.fit_transform(X_train)
    mean_before = normalizer._mean.copy()  # noqa: SLF001
    std_before = normalizer._std.copy()    # noqa: SLF001

    X_test_norm = normalizer.transform(X_test)

    assert X_train_norm.shape == X_train.shape
    assert X_test_norm.shape == X_test.shape

    assert np.allclose(X_train_norm.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_train_norm.std(axis=0), 1.0, atol=1e-7)

    assert np.array_equal(normalizer._mean, mean_before)  # noqa: SLF001
    assert np.array_equal(normalizer._std, std_before)    # noqa: SLF001

