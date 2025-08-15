from enum import StrEnum, auto
from typing import Literal

import numpy as np

import random_tree_models.utils as utils
from random_tree_models import rs_entropy, rs_gini_impurity


def check_y_and_target_groups(y: np.ndarray, target_groups: np.ndarray = None):
    n = len(y)
    if n == 0:
        raise ValueError(f"{n=}, expected at least one target value")

    if target_groups is not None and len(target_groups) != n:
        raise ValueError(f"{y.shape=} != {target_groups.shape=}")


def calc_variance(y: np.ndarray, target_groups: np.ndarray, **kwargs) -> float:
    """Calculates the variance of a split"""

    check_y_and_target_groups(y, target_groups=target_groups)

    n = len(y)

    if len(np.unique(target_groups)) == 1:
        return -np.var(y)

    w_left = target_groups.sum() / n
    w_right = 1.0 - w_left

    var_left = np.var(y[target_groups])
    var_right = np.var(y[~target_groups])

    var = w_left * var_left + w_right * var_right
    return -var


def entropy(y: np.ndarray) -> float:
    "Calculates the entropy across target values"

    n = len(y)
    check_y_and_target_groups(y)

    unique_ys = np.unique(y)
    ns = np.array([(y == y_val).sum() for y_val in unique_ys], dtype=int)

    ps = ns / float(n)

    if (ps == 1).any():
        return 0

    mask_ne0 = ~np.isclose(ps, 0)

    h = ps[mask_ne0] * np.log2(ps[mask_ne0])
    h = h.sum()

    return h


def calc_entropy(y: np.ndarray, target_groups: np.ndarray, **kwargs) -> float:
    """Calculates the entropy of a split"""

    check_y_and_target_groups(y, target_groups=target_groups)

    w_left = target_groups.sum() / len(target_groups)
    w_right = 1.0 - w_left

    h_left = entropy(y[target_groups]) if w_left > 0 else 0
    h_right = entropy(y[~target_groups]) if w_right > 0 else 0

    h = w_left * h_left + w_right * h_right
    return h


def calc_entropy_rs(y: np.ndarray, target_groups: np.ndarray, **kwargs) -> float:
    """Calculates the entropy of a split"""

    check_y_and_target_groups(y, target_groups=target_groups)

    w_left = target_groups.sum() / len(target_groups)
    w_right = 1.0 - w_left

    h_left = rs_entropy(y[target_groups]) if w_left > 0 else 0
    h_right = rs_entropy(y[~target_groups]) if w_right > 0 else 0

    h = w_left * h_left + w_right * h_right
    return h


def gini_impurity(y: np.ndarray) -> float:
    "Calculates the gini impurity across target values"

    check_y_and_target_groups(y)

    n = len(y)

    unique_ys = np.unique(y)
    ns = np.array([(y == y_val).sum() for y_val in unique_ys], dtype=int)

    ps = ns / float(n)

    if (ps == 1).any():
        return 0

    mask_ne0 = ~np.isclose(ps, 0)

    g = ps[mask_ne0] * (1 - ps[mask_ne0])
    g = g.sum()

    return -g


def calc_gini_impurity(y: np.ndarray, target_groups: np.ndarray, **kwargs) -> float:
    """Calculates the gini impurity of a split

    Based on: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
    """

    check_y_and_target_groups(y, target_groups=target_groups)

    w_left = target_groups.sum() / len(target_groups)
    w_right = 1.0 - w_left

    g_left = gini_impurity(y[target_groups]) if w_left > 0 else 0
    g_right = gini_impurity(y[~target_groups]) if w_right > 0 else 0

    g = w_left * g_left + w_right * g_right
    return g


def calc_gini_impurity_rs(y: np.ndarray, target_groups: np.ndarray, **kwargs) -> float:
    """Calculates the gini impurity of a split

    Based on: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
    """

    check_y_and_target_groups(y, target_groups=target_groups)

    w_left = target_groups.sum() / len(target_groups)
    w_right = 1.0 - w_left

    g_left = rs_gini_impurity(y[target_groups]) if w_left > 0 else 0
    g_right = rs_gini_impurity(y[~target_groups]) if w_right > 0 else 0

    g = w_left * g_left + w_right * g_right
    return g


def xgboost_split_score(
    g: np.ndarray, h: np.ndarray, growth_params: utils.TreeGrowthParameters
) -> float:
    "Equation 7 in Chen et al 2016, XGBoost: A Scalable Tree Boosting System"
    check_y_and_target_groups(g, target_groups=h)

    top = g.sum()
    top = top * top

    bottom = h.sum() + growth_params.lam

    if bottom == 0:
        return 0.0

    score = top / bottom
    return -score


def calc_xgboost_split_score(
    y: np.ndarray,
    target_groups: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    growth_params: utils.TreeGrowthParameters,
    **kwargs,
) -> float:
    """Calculates the xgboost general version score of a split with loss specifics in g and h.

    Based on: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
    """

    check_y_and_target_groups(g, target_groups=target_groups)
    check_y_and_target_groups(h, target_groups=target_groups)

    n_left = target_groups.sum()
    n_right = len(target_groups) - n_left

    score_left = (
        xgboost_split_score(g[target_groups], h[target_groups], growth_params)
        if n_left > 0
        else 0
    )
    score_right = (
        xgboost_split_score(g[~target_groups], h[~target_groups], growth_params)
        if n_right > 0
        else 0
    )

    score = score_left + score_right
    return score


class IncrementingScore:
    score = 0

    def __call__(self, *args, **kwargs) -> float:
        """Calculates the random cut score of a split"""
        self.score += 1
        return self.score


INC_SCORE = 0


def calc_incrementing_score() -> float:
    global INC_SCORE
    INC_SCORE += 1
    return INC_SCORE


def reset_incrementing_score():
    global INC_SCORE
    INC_SCORE = 0


class MetricNames(StrEnum):
    variance = auto()
    entropy = auto()
    entropy_rs = auto()
    gini = auto()
    gini_rs = auto()
    # variance for split score because Friedman et al. 2001 in Algorithm 1
    # step 4 minimize the squared error between actual and predicted dloss/dyhat
    friedman_binary_classification = auto()
    xgboost = auto()
    incrementing = auto()


MetricNameLiterals = Literal[
    "variance",
    "entropy",
    "entropy_rs",
    "gini",
    "gini_rs",
    "friedman_binary_classification",
    "xgboost",
    "incrementing",
]


def calc_split_score(
    metric: MetricNames,
    y: np.ndarray,
    target_groups: np.ndarray,
    yhat: np.ndarray | None = None,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    growth_params: utils.TreeGrowthParameters | None = None,
) -> float:
    match metric:
        case MetricNames.variance:
            return calc_variance(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.entropy:
            return calc_entropy(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.entropy_rs:
            return calc_entropy_rs(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.gini:
            return calc_gini_impurity(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.gini_rs:
            return calc_gini_impurity_rs(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.friedman_binary_classification:
            return calc_variance(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.xgboost:
            return calc_xgboost_split_score(
                y, target_groups, yhat=yhat, g=g, h=h, growth_params=growth_params
            )
        case MetricNames.incrementing:
            return calc_incrementing_score()
        case _:
            raise NotImplementedError(
                f"calc_split_score is not implemented for {metric=}."
            )
