import numpy as np

import random_tree_models.scoring as scoring
import random_tree_models.utils as utils


def leaf_weight_mean(y: np.ndarray, **kwargs) -> float:
    return np.mean(y)


def leaf_weight_binary_classification_friedman2001(
    g: np.ndarray,
    **kwargs,
) -> float:
    "Computes optimal leaf weight as in Friedman et al. 2001 Algorithm 5"

    g_abs = np.abs(g)
    gamma_jm = g.sum() / (g_abs * (2 - g_abs)).sum()
    return gamma_jm


def leaf_weight_xgboost(
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray,
    h: np.ndarray,
    **kwargs,
) -> float:
    "Computes optimal leaf weight as in Chen et al. 2016 equation 5"

    w = -g.sum() / (h + growth_params.lam).sum()
    return w


def calc_leaf_weight(
    y: np.ndarray,
    measure_name: scoring.SplitScoreMetrics,
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray = None,
    h: np.ndarray = None,
) -> float:
    """Calculates the leaf weight, depending on the choice of measure_name.

    This computation assumes all y values are part of the same leaf.
    """

    if len(y) == 0:
        return None

    match measure_name:
        case (
            scoring.SplitScoreMetrics.variance
            | scoring.SplitScoreMetrics.entropy
            | scoring.SplitScoreMetrics.entropy_rs
            | scoring.SplitScoreMetrics.gini
            | scoring.SplitScoreMetrics.gini_rs
            | scoring.SplitScoreMetrics.incrementing
        ):
            leaf_weight = leaf_weight_mean(y)
        case scoring.SplitScoreMetrics.friedman_binary_classification:
            leaf_weight = leaf_weight_binary_classification_friedman2001(g)
        case scoring.SplitScoreMetrics.xgboost:
            leaf_weight = leaf_weight_xgboost(growth_params, g, h)
        case _:
            raise KeyError(
                f"Unknown measure_name: {measure_name}, expected one of {', '.join(list(scoring.SplitScoreMetrics.__members__.keys()))}"
            )

    return leaf_weight
