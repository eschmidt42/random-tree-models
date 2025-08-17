import numpy as np

import random_tree_models.params as utils
from random_tree_models.params import MetricNames


def leaf_weight_mean(y: np.ndarray) -> float:
    return float(np.mean(y))


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
    measure_name: utils.MetricNames,
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> float | None:
    """Calculates the leaf weight, depending on the choice of measure_name.

    This computation assumes all y values are part of the same leaf.
    """

    if len(y) == 0:
        return None

    match measure_name:
        case (
            MetricNames.variance
            | MetricNames.entropy
            | MetricNames.entropy_rs
            | MetricNames.gini
            | MetricNames.gini_rs
            | MetricNames.incrementing
        ):
            return leaf_weight_mean(y)
        case MetricNames.friedman_binary_classification:
            if g is None:
                raise ValueError(f"{g=} cannot be None for {measure_name=}")
            return leaf_weight_binary_classification_friedman2001(g)
        case MetricNames.xgboost:
            if g is None or h is None:
                raise ValueError(f"{g=} and {h=} cannot be None for {measure_name=}")
            return leaf_weight_xgboost(growth_params, g, h)
        case _:
            raise NotImplementedError(
                f"calc_split_score is not implemented for {measure_name=}."
            )
