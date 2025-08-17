import numpy as np
import pytest

import random_tree_models.leafweights as leafweights
import random_tree_models.params as utils
from random_tree_models.params import MetricNames


def test_leaf_weight_mean():
    y = np.array([1, 2, 3])
    assert leafweights.leaf_weight_mean(y=y) == 2.0


def test_leaf_weight_binary_classification_friedman2001():
    g = np.array([1, 2, 3]) * 2
    assert leafweights.leaf_weight_binary_classification_friedman2001(g=g) == -0.375


def test_leaf_weight_xgboost():
    g = np.array([1, 2, 3]) * 2
    h = np.array([1, 2, 3]) * 4
    params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
    assert leafweights.leaf_weight_xgboost(g=g, h=h, growth_params=params) == -0.5


class Test_calc_leaf_weight:
    def test_error_for_unknown_scheme(self):
        y = np.array([1, 2, 3])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
        with pytest.raises(NotImplementedError):
            leafweights.calc_leaf_weight(
                y=y,
                growth_params=growth_params,
                measure_name="not_a_scheme",  # type: ignore
            )

    def test_leaf_weight_none_if_y_empty(self):
        y = np.array([])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)

        weight = leafweights.calc_leaf_weight(
            y=y, growth_params=growth_params, measure_name=MetricNames.gini
        )
        assert weight is None

    def test_leaf_weight_float_if_y_not_empty(self):
        y = np.array([1, 2, 3])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)

        weight = leafweights.calc_leaf_weight(
            y=y, growth_params=growth_params, measure_name=MetricNames.variance
        )
        assert isinstance(weight, float)
