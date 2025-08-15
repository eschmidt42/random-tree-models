from enum import Enum
from functools import partial

import numpy as np

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


class LeafWeightSchemes(Enum):
    # https://stackoverflow.com/questions/40338652/how-to-define-enum-values-that-are-functions
    friedman_binary_classification = partial(
        leaf_weight_binary_classification_friedman2001
    )
    variance = partial(leaf_weight_mean)
    entropy = partial(leaf_weight_mean)
    entropy_rs = partial(leaf_weight_mean)
    gini = partial(leaf_weight_mean)
    gini_rs = partial(leaf_weight_mean)
    xgboost = partial(leaf_weight_xgboost)
    incrementing = partial(leaf_weight_mean)

    def __call__(
        self,
        y: np.ndarray,
        growth_params: utils.TreeGrowthParameters,
        g: np.ndarray = None,
        h: np.ndarray = None,
    ) -> float:
        return self.value(y=y, growth_params=growth_params, g=g, h=h)


def calc_leaf_weight(
    y: np.ndarray,
    measure_name: str,
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray = None,
    h: np.ndarray = None,
) -> float:
    """Calculates the leaf weight, depending on the choice of measure_name.

    This computation assumes all y values are part of the same leaf.
    """

    if len(y) == 0:
        return None

    weight_func = LeafWeightSchemes[measure_name]
    leaf_weight = weight_func(y=y, growth_params=growth_params, g=g, h=h)

    return leaf_weight
