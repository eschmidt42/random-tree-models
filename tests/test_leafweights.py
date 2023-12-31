# -*- coding: utf-8 -*-
import numpy as np
import pytest
import pydantic

import random_tree_models.leafweights as leafweights
import random_tree_models.utils as utils


def test_leaf_weight_mean():
    y = np.array([1, 2, 3])
    assert leafweights.leaf_weight_mean(y=y) == 2.0


def test_leaf_weight_binary_classification_friedman2001():
    g = np.array([1, 2, 3]) * 2
    assert (
        leafweights.leaf_weight_binary_classification_friedman2001(g=g)
        == -0.375
    )


def test_leaf_weight_xgboost():
    g = np.array([1, 2, 3]) * 2
    h = np.array([1, 2, 3]) * 4
    params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
    assert (
        leafweights.leaf_weight_xgboost(growth_params=params, g=g, h=h) == -0.5
    )


class Test_calc_leaf_weight:
    def test_error_for_unknown_scheme(self):
        try:
            _ = utils.TreeGrowthParameters(
                max_depth=2, split_score_metric="not_a_scheme", lam=0.0
            )
        except pydantic.ValidationError:
            pytest.xfail("ValueError correctly raised for unknown scheme")
        else:
            pytest.fail("ValueError not raised for unknown scheme")

    # returns a float if y is not empty
    def test_leaf_weight_float_if_y_not_empty(self):
        y = np.array([1, 2, 3])
        growth_params = utils.TreeGrowthParameters(
            max_depth=2,
            split_score_metric=utils.SplitScoreMetrics["variance"],
            lam=0.0,
        )

        weight = leafweights.calc_leaf_weight(
            y=y,
            growth_params=growth_params,
        )
        assert isinstance(weight, float)
