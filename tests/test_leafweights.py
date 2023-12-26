import numpy as np
import pytest

import random_tree_models.leafweights as leafweights
import random_tree_models.scoring as scoring
import random_tree_models.utils as utils


def test_leaf_weight_mean():
    y = np.array([1, 2, 3])
    g = np.array([1, 2, 3]) * 2
    assert leafweights.leaf_weight_mean(y=y, g=g) == 2.0


def test_leaf_weight_binary_classification_friedman2001():
    y = np.array([1, 2, 3])
    g = np.array([1, 2, 3]) * 2
    assert (
        leafweights.leaf_weight_binary_classification_friedman2001(y=y, g=g)
        == -0.375
    )


def test_leaf_weight_xgboost():
    y = np.array([1, 2, 3])
    g = np.array([1, 2, 3]) * 2
    h = np.array([1, 2, 3]) * 4
    params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
    assert (
        leafweights.leaf_weight_xgboost(y=y, g=g, h=h, growth_params=params)
        == -0.5
    )


class Test_calc_leaf_weight:
    def test_error_for_unknown_scheme(self):
        y = np.array([1, 2, 3])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
        try:
            leafweights.calc_leaf_weight(
                y=y, growth_params=growth_params, measure_name="not_a_scheme"
            )
        except KeyError:
            pytest.xfail("ValueError correctly raised for unknown scheme")
        else:
            pytest.fail("ValueError not raised for unknown scheme")

    def test_leaf_weight_none_if_y_empty(self):
        y = np.array([])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)

        weight = leafweights.calc_leaf_weight(
            y=y, growth_params=growth_params, measure_name="not_a_scheme"
        )
        assert weight is None

    # returns a float if y is not empty
    def test_leaf_weight_float_if_y_not_empty(self):
        y = np.array([1, 2, 3])
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)

        weight = leafweights.calc_leaf_weight(
            y=y,
            growth_params=growth_params,
            measure_name=scoring.SplitScoreMetrics["variance"],
        )
        assert isinstance(weight, float)
