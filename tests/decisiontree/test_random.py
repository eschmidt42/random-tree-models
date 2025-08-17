import numpy as np

from random_tree_models.decisiontree.random import (
    get_random_feature_ids,
    get_random_sample_ids,
)


def test_get_random_sample_ids():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    rng = np.random.RandomState(42)

    # Test with frac_subsamples = 1.0
    ix_samples_full = get_random_sample_ids(X, rng, 1.0)
    assert np.array_equal(ix_samples_full, np.array([0, 1, 2, 3, 4]))
    assert len(set(ix_samples_full)) == len(ix_samples_full)

    # Test with frac_subsamples < 1.0
    ix_samples_partial = get_random_sample_ids(X, rng, 0.5)
    assert len(ix_samples_partial) == int(0.5 * len(X))
    assert all(i in np.arange(len(X)) for i in ix_samples_partial)
    assert len(set(ix_samples_partial)) == len(ix_samples_partial)


def test_get_random_feature_ids():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    rng = np.random.RandomState(42)

    # Test with frac_features = 1.0
    ix_features_full = get_random_feature_ids(X, rng, 1.0)
    assert np.array_equal(ix_features_full, np.array([0, 1, 2]))
    assert len(set(ix_features_full)) == len(ix_features_full)

    # Test with frac_features < 1.0
    ix_features_partial = get_random_feature_ids(X, rng, 0.5)
    assert len(ix_features_partial) == int(X.shape[1] * 0.5)
    assert all(i in np.arange(X.shape[1]) for i in ix_features_partial)
    assert len(set(ix_features_partial)) == len(ix_features_partial)
