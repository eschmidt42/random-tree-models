import numpy as np
import pytest

import random_tree_models.scoring as scoring
import random_tree_models.utils as utils
from random_tree_models import rs_entropy, rs_gini_impurity


@pytest.mark.parametrize(
    "y, target_groups, is_bad",
    [
        (np.array([]), np.array([False, True]), True),
        (np.array([]), np.array([False]), True),
        (np.array([]), np.array([]), True),
        (np.array([]), None, True),
        (np.array([1]), np.array([False, True]), True),
        (np.array([1, 0]), np.array([False, True]), False),
        (np.array([1, 1]), np.array([True]), True),
    ],
)
def test_check_y_and_target_groups(y, target_groups, is_bad):
    try:
        # line to test
        scoring.check_y_and_target_groups(y, target_groups=target_groups)
    except ValueError as ex:
        if is_bad:
            pytest.xfail("y and target_groups properly failed")
        else:
            raise ex
    else:
        if is_bad:
            pytest.fail(f"{y=} {target_groups=} should have failed but didn't")


@pytest.mark.parametrize(
    "y, target_groups, variance_exp",
    [
        (np.array([]), None, None),
        (np.array([1]), np.array([False, True]), None),
        (np.array([1]), np.array([True]), 0),
        (np.array([1, 1]), np.array([True, True]), 0),
        (np.array([1]), np.array([False]), 0),
        (np.array([1, 1]), np.array([False, False]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, False, True, True]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, True, False, True]), -0.25),
    ],
)
def test_calc_variance(
    y: np.ndarray, target_groups: np.ndarray, variance_exp: float
):
    try:
        # line to test
        variance = scoring.calc_variance(y, target_groups)
    except ValueError as ex:
        if variance_exp is None:
            pytest.xfail("Properly raised error calculating the variance")
        else:
            raise ex
    else:
        if variance_exp is None:
            pytest.fail("calc_variance should have failed but didn't")
        assert variance == variance_exp


@pytest.mark.parametrize(
    "y",
    [
        np.array([]),
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_entropy(y: np.ndarray):
    try:
        # line to test
        h = scoring.entropy(y)
    except ValueError as ex:
        if len(y) == 0:
            pytest.xfail("entropy properly failed because of empty y")
        else:
            raise ex
    else:
        if len(y) == 0:
            pytest.fail("entropy should have failed but didn't")

        assert np.less_equal(h, 0)


@pytest.mark.parametrize(
    "y",
    [
        np.array([]),
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_entropy_rs(y: np.ndarray):
    try:
        # line to test
        h = rs_entropy(y)
    except ValueError as ex:
        if len(y) == 0:
            pytest.xfail("entropy properly failed because of empty y")
        else:
            raise ex
    else:
        if len(y) == 0:
            pytest.fail("entropy should have failed but didn't")

        assert np.less_equal(h, 0)


@pytest.mark.parametrize(
    "y, target_groups, h_exp",
    [
        (np.array([]), None, None),
        (np.array([1]), np.array([False, True]), None),
        (np.array([1]), np.array([True]), 0),
        (np.array([1, 1]), np.array([True, True]), 0),
        (np.array([1]), np.array([False]), 0),
        (np.array([1, 1]), np.array([False, False]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, False, True, True]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, True, False, True]), -1.0),
    ],
)
def test_calc_entropy(y: np.ndarray, target_groups: np.ndarray, h_exp: float):
    try:
        # line to test
        h = scoring.calc_entropy(y, target_groups)
    except ValueError as ex:
        if h_exp is None:
            pytest.xfail("Properly raised error calculating the entropy")
        else:
            raise ex
    else:
        if h_exp is None:
            pytest.fail("calc_entropy should have failed but didn't")
        assert h == h_exp


@pytest.mark.parametrize(
    "y, target_groups, h_exp",
    [
        (np.array([]), None, None),
        (np.array([1]), np.array([False, True]), None),
        (np.array([1]), np.array([True]), 0),
        (np.array([1, 1]), np.array([True, True]), 0),
        (np.array([1]), np.array([False]), 0),
        (np.array([1, 1]), np.array([False, False]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, False, True, True]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, True, False, True]), -1.0),
    ],
)
def test_calc_entropy_rs(
    y: np.ndarray, target_groups: np.ndarray, h_exp: float
):
    try:
        # line to test
        h = scoring.calc_entropy_rs(y, target_groups)
    except ValueError as ex:
        if h_exp is None:
            pytest.xfail("Properly raised error calculating the entropy")
        else:
            raise ex
    else:
        if h_exp is None:
            pytest.fail("calc_entropy should have failed but didn't")
        assert h == h_exp


@pytest.mark.parametrize(
    "y",
    [
        np.array([]),
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_gini_impurity(y: np.ndarray):
    try:
        # line to test
        g = scoring.gini_impurity(y)
    except ValueError as ex:
        if len(y) == 0:
            pytest.xfail("gini_impurity properly failed because of empty y")
        else:
            raise ex
    else:
        if len(y) == 0:
            pytest.fail("gini_impurity should have failed but didn't")

        assert np.less_equal(g, 0)


@pytest.mark.parametrize(
    "y",
    [
        np.array([]),
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_gini_impurity_rs(y: np.ndarray):
    try:
        # line to test
        g = rs_gini_impurity(y)
    except ValueError as ex:
        if len(y) == 0:
            pytest.xfail("gini_impurity properly failed because of empty y")
        else:
            raise ex
    else:
        if len(y) == 0:
            pytest.fail("gini_impurity should have failed but didn't")

        assert np.less_equal(g, 0)


@pytest.mark.parametrize(
    "y",
    [
        np.array([1]),
        np.array([1, 2]),
    ],
)
def test_gini_impurity_py_vs_rs(y: np.ndarray):
    g_py = scoring.gini_impurity(y)
    g_rs = rs_gini_impurity(y)

    assert np.isclose(g_py, g_rs)


@pytest.mark.parametrize(
    "y, target_groups, g_exp",
    [
        (np.array([]), None, None),
        (np.array([1]), np.array([False, True]), None),
        (np.array([1]), np.array([True]), 0),
        (np.array([1, 1]), np.array([True, True]), 0),
        (np.array([1]), np.array([False]), 0),
        (np.array([1, 1]), np.array([False, False]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, False, True, True]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, True, False, True]), -0.5),
    ],
)
def test_calc_gini_impurity(
    y: np.ndarray, target_groups: np.ndarray, g_exp: float
):
    try:
        # line to test
        g = scoring.calc_gini_impurity(y, target_groups)
    except ValueError as ex:
        if g_exp is None:
            pytest.xfail("Properly raised error calculating the gini impurity")
        else:
            raise ex
    else:
        if g_exp is None:
            pytest.fail("calc_gini_impurity should have failed but didn't")
        assert g == g_exp


@pytest.mark.parametrize(
    "y, target_groups, g_exp",
    [
        (np.array([]), None, None),
        (np.array([1]), np.array([False, True]), None),
        (np.array([1]), np.array([True]), 0),
        (np.array([1, 1]), np.array([True, True]), 0),
        (np.array([1]), np.array([False]), 0),
        (np.array([1, 1]), np.array([False, False]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, False, True, True]), 0),
        (np.array([1, 1, 2, 2]), np.array([False, True, False, True]), -0.5),
    ],
)
def test_calc_gini_impurity_rs(
    y: np.ndarray, target_groups: np.ndarray, g_exp: float
):
    try:
        # line to test
        g = scoring.calc_gini_impurity_rs(y, target_groups)
    except ValueError as ex:
        if g_exp is None:
            pytest.xfail("Properly raised error calculating the gini impurity")
        else:
            raise ex
    else:
        if g_exp is None:
            pytest.fail("calc_gini_impurity should have failed but didn't")
        assert g == g_exp


@pytest.mark.parametrize(
    "g,h,is_bad",
    [
        (np.array([]), np.array([]), True),
        (np.array([1]), np.array([1]), False),
        (np.array([1]), np.array([1, 2]), True),
        (np.array([1, 2]), np.array([1]), True),
        (np.array([1, 2]), np.array([1, 2]), False),
    ],
)
def test_xgboost_split_score(g: np.ndarray, h: np.ndarray, is_bad: bool):
    growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
    try:
        # line to test
        score = scoring.xgboost_split_score(g, h, growth_params)
    except ValueError as ex:
        if is_bad:
            pytest.xfail(
                "xgboost_split_score properly failed because of empty g or h"
            )
        else:
            raise ex
    else:
        if is_bad:
            pytest.fail(
                "xgboost_split_score should have failed due to empty g or h but didn't"
            )

        assert np.less_equal(score, 0)


@pytest.mark.parametrize(
    "g, h, target_groups, score_exp",
    [
        # failure cases
        (np.array([]), np.array([]), np.array([]), None),
        (np.array([1]), np.array([-1.0]), np.array([False, True]), None),
        # regression cases - least squares - no split worse than with split
        (
            np.array([-0.5, 0.5]),
            np.array([-1.0, -1.0]),
            np.array([True, False]),
            0.5,
        ),  # with split
        (
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
            np.array([False, False]),
            0.0,
        ),  # without split
        # classification cases - binomial log-likelihood
        (
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
            np.array([True, False]),
            2.0,
        ),  # with split
        (
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
            np.array([False, False]),
            0.0,
        ),  # without split
    ],
)
def test_calc_xgboost_split_score(
    g: np.ndarray, h: np.ndarray, target_groups: np.ndarray, score_exp: float
):
    growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
    y = None
    try:
        # line to test
        score = scoring.calc_xgboost_split_score(
            y, target_groups, g, h, growth_params
        )
    except ValueError as ex:
        if score_exp is None:
            pytest.xfail("Properly raised error calculating the xgboost score")
        else:
            raise ex
    else:
        if score_exp is None:
            pytest.fail(
                "calc_xgboost_split_score should have failed but didn't"
            )
        assert score == score_exp


class TestSplitScoreMetrics:
    "Redudancy test - calling calc_xgboost_split_score etc via SplitScoreMetrics needs to yield the same values as in the test above."
    y = np.array([1, 1, 2, 2])
    target_groups = np.array([False, True, False, True])

    g_exp = -0.5
    h_exp = -1.0
    var_exp = -0.25

    def test_gini(self):
        g = scoring.SplitScoreMetrics["gini"](self.y, self.target_groups)
        assert g == self.g_exp

    def test_gini_rs(self):
        g = scoring.SplitScoreMetrics["gini_rs"](self.y, self.target_groups)
        assert g == self.g_exp

    def test_entropy(self):
        h = scoring.SplitScoreMetrics["entropy"](self.y, self.target_groups)
        assert h == self.h_exp

    def test_entropy(self):
        h = scoring.SplitScoreMetrics["entropy_rs"](self.y, self.target_groups)
        assert h == self.h_exp

    def test_variance(self):
        var = scoring.SplitScoreMetrics["variance"](self.y, self.target_groups)
        assert var == self.var_exp

    def test_friedman_binary_classification(self):
        var = scoring.SplitScoreMetrics["friedman_binary_classification"](
            self.y, self.target_groups
        )
        assert var == self.var_exp

    @pytest.mark.parametrize(
        "g, h, target_groups, score_exp",
        [
            # regression cases - least squares - no split worse than with split
            (
                np.array([-0.5, 0.5]),
                np.array([-1.0, -1.0]),
                np.array([True, False]),
                0.5,
            ),  # with split
            (
                np.array([-1.0, 1.0]),
                np.array([-1.0, -1.0]),
                np.array([False, False]),
                0.0,
            ),  # without split
            # classification cases - binomial log-likelihood
            (
                np.array([-1.0, 1.0]),
                np.array([-1.0, -1.0]),
                np.array([True, False]),
                2.0,
            ),  # with split
            (
                np.array([-1.0, 1.0]),
                np.array([-1.0, -1.0]),
                np.array([False, False]),
                0.0,
            ),  # without split
        ],
    )
    def test_xgboost(
        self,
        g: np.ndarray,
        h: np.ndarray,
        target_groups: np.ndarray,
        score_exp: float,
    ):
        growth_params = utils.TreeGrowthParameters(max_depth=2, lam=0.0)
        y = None

        # line to test
        score = scoring.calc_xgboost_split_score(
            y, target_groups, g, h, growth_params
        )

        assert score == score_exp
