# -*- coding: utf-8 -*-
import numpy as np

import random_tree_models.utils as utils


def leaf_weight_mean(y: np.ndarray) -> float:
    return np.mean(y)


def leaf_weight_binary_classification_friedman2001(
    g: np.ndarray,
) -> float:
    "Computes optimal leaf weight as in Friedman et al. 2001 Algorithm 5"

    g_abs = np.abs(g)
    gamma_jm = g.sum() / (g_abs * (2 - g_abs)).sum()
    return gamma_jm


def leaf_weight_xgboost(
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray,
    h: np.ndarray,
) -> float:
    "Computes optimal leaf weight as in Chen et al. 2016 equation 5"

    w = -g.sum() / (h + growth_params.lam).sum()
    return w


def calc_leaf_weight(
    y: np.ndarray,
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray = None,
    h: np.ndarray = None,
) -> float:
    """Calculates the leaf weight, depending on the choice of measure_name.

    This computation assumes all y values are part of the same leaf.
    """

    if len(y) == 0:
        return None

    measure_name = growth_params.split_score_metric

    match measure_name:
        case (
            utils.SplitScoreMetrics.variance
            | utils.SplitScoreMetrics.entropy
            | utils.SplitScoreMetrics.entropy_rs
            | utils.SplitScoreMetrics.gini
            | utils.SplitScoreMetrics.gini_rs
            | utils.SplitScoreMetrics.incrementing
        ):
            leaf_weight = leaf_weight_mean(y)
        case utils.SplitScoreMetrics.friedman_binary_classification:
            leaf_weight = leaf_weight_binary_classification_friedman2001(g)
        case utils.SplitScoreMetrics.xgboost:
            leaf_weight = leaf_weight_xgboost(growth_params, g, h)
        case _:
            raise KeyError(
                f"Unknown measure_name: {measure_name}, expected one of {', '.join(list(utils.SplitScoreMetrics.__members__.keys()))}"
            )

    return leaf_weight
