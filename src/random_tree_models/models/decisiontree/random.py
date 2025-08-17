import numpy as np


def get_random_sample_ids(
    X: np.ndarray, rng: np.random.RandomState, frac_subsamples: float
) -> np.ndarray:
    ix = np.arange(len(X))
    if frac_subsamples < 1.0:
        n_samples = int(frac_subsamples * len(X))
        ix_samples = rng.choice(ix, size=n_samples, replace=False)
    else:
        ix_samples = ix
    return ix_samples


def get_random_feature_ids(
    X: np.ndarray, rng: np.random.RandomState, frac_features: float
) -> np.ndarray:
    if frac_features < 1.0:
        n_columns = int(X.shape[1] * frac_features)
        ix_features = rng.choice(
            np.arange(X.shape[1]),
            size=n_columns,
            replace=False,
        )
    else:
        ix_features = np.arange(X.shape[1])
    return ix_features
