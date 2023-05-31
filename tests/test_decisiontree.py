import numpy as np
import pytest
from pydantic import ValidationError

import random_tree_models.decisiontree as dtree

# first value in each tuple is the value to test and the second is the flag indicating if this should work
INT_OPTIONS_NONE_OKAY = [(0, True), (None, True), ("blub", False)]
INT_OPTIONS_NONE_NOT_OKAY = [(0, True), (None, False), ("blub", False)]
FLOAT_OPTIONS_NONE_OKAY = [
    (-1.0, True),
    (None, True),
    ("blub", False),
]
FLOAT_OPTIONS_NONE_NOT_OKAY = [
    (-1.0, True),
    (None, False),
    ("blub", False),
]
NODE_OPTIONS_NONE_OKAY = [
    (dtree.Node(), True),
    (None, True),
    ("blub", False),
]
STR_OPTIONS_NONE_OKAY = [("blub", True), (None, True), (1.0, False)]
STR_OPTIONS_NONE_NOT_OKAY = [
    ("blub", True),
    (None, False),
    (1, False),
    (1.0, False),
]


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in STR_OPTIONS_NONE_NOT_OKAY
        for value in FLOAT_OPTIONS_NONE_OKAY
    ],
)
def test_SplitScore(name, value):
    name, name_okay = name
    value, value_okay = value
    is_okay = name_okay and value_okay
    is_bad = not is_okay
    try:
        # line to test
        measure = dtree.SplitScore(name=name, value=value)
    except ValidationError as ex:
        if is_okay:
            raise ValueError(f"whoops {name=} {value=} failed with {ex}")
        else:
            pytest.xfail("SplitScore validation failed as expected")
    else:
        if is_bad:
            pytest.fail(
                f"SplitScore test unexpectedly passed for {name=}, {value=}, {name_okay=}, {value_okay=}, {is_okay=}"
            )

        assert hasattr(measure, "name")
        assert hasattr(measure, "value")


@pytest.mark.parametrize(
    "int_val, float_val, node_val, str_val",
    [
        (int_val, float_val, node_val, str_val)
        for int_val in INT_OPTIONS_NONE_OKAY
        for float_val in FLOAT_OPTIONS_NONE_OKAY
        for node_val in NODE_OPTIONS_NONE_OKAY
        for str_val in STR_OPTIONS_NONE_OKAY
    ],
)
def test_Node(int_val, float_val, node_val, str_val):
    array_column, array_column_okay = int_val
    threshold, threshold_okay = float_val
    prediction, prediction_okay = float_val
    left, left_okay = node_val
    right, right_okay = node_val
    n_obs, n_obs_okay = int_val
    reason, reason_okay = str_val

    is_okay = all(
        [
            array_column_okay,
            threshold_okay,
            prediction_okay,
            left_okay,
            right_okay,
            n_obs_okay,
            reason_okay,
        ]
    )
    measure = dtree.SplitScore(name="blub", value=1.0)
    try:
        # line to test
        node = dtree.Node(
            array_column=array_column,
            threshold=threshold,
            prediction=prediction,
            left=left,
            right=right,
            measure=measure,
            n_obs=n_obs,
            reason=reason,
        )
    except ValidationError as ex:
        if is_okay:
            raise ex
        else:
            pytest.xfail("SplitScore validation failed as expected")
    else:
        for att in [
            "array_column",
            "threshold",
            "prediction",
            "left",
            "right",
            "measure",
            "n_obs",
            "reason",
            "node_id",
        ]:
            assert hasattr(node, att), f"{att=} missing in Node"
        assert node.is_leaf == (
            (left is None) and (right is None)
        ), f"left: {left is None} right: {right is None}"


@pytest.mark.parametrize(
    "y, depths",
    [
        (y, depths)
        for y in [(np.array([1, 2]), False), (np.array([]), True)]
        for depths in [(1, 2, False), (2, 2, True), (3, 2, True)]
    ],
)
def test_check_is_baselevel(y, depths):
    node = dtree.Node()

    y, is_baselevel_exp_y = y
    depth, max_depth, is_baselevel_exp_depth = depths
    is_baselevel_exp = is_baselevel_exp_depth or is_baselevel_exp_y

    # line to test
    is_baselevel, msg = dtree.check_is_baselevel(
        y, node, depth=depth, max_depth=max_depth
    )

    assert is_baselevel == is_baselevel_exp
    assert isinstance(msg, str)


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
        dtree.check_y_and_target_groups(y, target_groups=target_groups)
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
        variance = dtree.calc_variance(y, target_groups)
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
        h = dtree.entropy(y)
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
        h = dtree.calc_entropy(y, target_groups)
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
        g = dtree.gini_impurity(y)
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
        g = dtree.calc_gini_impurity(y, target_groups)
    except ValueError as ex:
        if g_exp is None:
            pytest.xfail("Properly raised error calculating the gini impurity")
        else:
            raise ex
    else:
        if g_exp is None:
            pytest.fail("calc_gini_impurity should have failed but didn't")
        assert g == g_exp


class TestSplitScoreMetrics:
    y = np.array([1, 1, 2, 2])
    target_groups = np.array([False, True, False, True])

    g_exp = -0.5
    h_exp = -1.0
    var_exp = -0.25

    def test_gini(self):
        g = dtree.SplitScoreMetrics["gini"](self.y, self.target_groups)
        assert g == self.g_exp

    def test_entropy(self):
        h = dtree.SplitScoreMetrics["entropy"](self.y, self.target_groups)
        assert h == self.h_exp

    def test_variance(self):
        var = dtree.SplitScoreMetrics["variance"](self.y, self.target_groups)
        assert var == self.var_exp


@pytest.mark.parametrize(
    "score,column,threshold,target_groups",
    [
        (score, column, threshold, target_groups)
        for score in FLOAT_OPTIONS_NONE_NOT_OKAY
        for column in INT_OPTIONS_NONE_NOT_OKAY
        for threshold in FLOAT_OPTIONS_NONE_NOT_OKAY
        for target_groups in [
            (np.array([1, 2, 3]), True),
            (np.array([]), True),
            (None, False),
        ]
    ],
)
def test_bestsplit(score, column, threshold, target_groups):
    score, score_okay = score
    column, column_okay = column
    threshold, threshold_okay = threshold
    target_groups, target_groups_okay = target_groups

    is_okay = all([score_okay, column_okay, threshold_okay, target_groups_okay])
    is_bad = not is_okay

    try:
        # line to test
        best = dtree.BestSplit(
            score=score,
            column=column,
            threshold=threshold,
            target_groups=target_groups,
        )
    except ValidationError as ex:
        if is_okay:
            raise ex
        else:
            pytest.xfail("BestSplit validation failed as expected")
    else:
        if is_bad:
            pytest.fail(
                f"BestSplit validation did pass unexpectedly with {score=}, {column=}, {threshold=}, {target_groups=}, {score_okay=}, {column_okay=}, {threshold_okay=}, {target_groups_okay=}, {is_bad=}"
            )

        assert hasattr(best, "score")
        assert hasattr(best, "column")
        assert hasattr(best, "threshold")
        assert hasattr(best, "target_groups")


class Test_find_best_split:
    """
    cases to test for all measure_name values:
    * simple & 1d is split as expected
        * classification: y = 1 class, y = 2 classes, y = 3 classes
        * regression: y = 1 value, y = 2 values, y = 3 values where 2 are more similar
    * simple & 2d is split as expected
        * same as 1d but 1st column useless and 2nd contains the needed info
    """

    X_1D = np.array(
        [
            [
                1,
            ],
            [
                2,
            ],
            [
                3,
            ],
            [
                4,
            ],
        ]
    )

    X_2D = np.hstack((np.ones_like(X_1D), X_1D))

    y_1class = np.ones(X_1D.shape[0], dtype=bool)
    y_2class = np.array([False, False, True, True])
    y_3class = np.array([0, 0, 1, 2])

    y_1reg = np.ones(X_1D.shape[0])
    y_2reg = np.array([-1.0, -1.0, 1.0, 1.0])
    y_3reg = np.array([-1.0, -0.9, 1.0, 2.0])

    @pytest.mark.parametrize(
        "y,ix,measure_name",
        [
            (y_1class, None, "gini"),
            (y_2class, 2, "gini"),
            (y_3class, 2, "gini"),
            (y_1class, None, "entropy"),
            (y_2class, 2, "entropy"),
            (y_3class, 2, "entropy"),
            (y_1reg, None, "variance"),
            (y_2reg, 2, "variance"),
            (y_3reg, 2, "variance"),
        ],
    )
    def test_1d(self, y: np.ndarray, ix: int, measure_name: str):
        is_homogenous = len(np.unique(y)) == 1
        try:
            # line to test
            best = dtree.find_best_split(
                self.X_1D, y, measure_name=measure_name
            )
        except ValueError as ex:
            if is_homogenous:
                pytest.xfail("Splitting a homogneous y failed as expected")
            else:
                raise ex
        else:
            if is_homogenous:
                pytest.fail("Splitting a homogneous y passed unexpectedly")

            threshold_exp = float(self.X_1D[ix, 0])
            assert best.threshold == threshold_exp

    @pytest.mark.parametrize(
        "y,ix,measure_name",
        [
            (y_1class, None, "gini"),
            (y_2class, 2, "gini"),
            (y_3class, 2, "gini"),
            (y_1class, None, "entropy"),
            (y_2class, 2, "entropy"),
            (y_3class, 2, "entropy"),
            (y_1reg, None, "variance"),
            (y_2reg, 2, "variance"),
            (y_3reg, 2, "variance"),
        ],
    )
    def test_2d(self, y: np.ndarray, ix: int, measure_name: str):
        is_homogenous = len(np.unique(y)) == 1
        try:
            # line to test
            best = dtree.find_best_split(self.X_2D, y, measure_name)
        except ValueError as ex:
            if is_homogenous:
                pytest.xfail("Splitting a homogneous y failed as expected")
            else:
                raise ex
        else:
            if is_homogenous:
                pytest.fail("Splitting a homogneous y passed unexpectedly")

            assert best.column == 1
            threshold_exp = float(self.X_2D[ix, 1])
            assert best.threshold == threshold_exp


# TODO: test grow_tree
def test_grow_tree():
    ...


# TODO: test check_if_has_prediction
def test_check_if_has_prediction():
    ...


# TODO: test predict_with_tree
def test_predict_with_tree():
    ...


# TODO: test DecisionTreeTemplate
def test_decisiontreetemplate():
    ...


# TODO: test DecisionTreeRegressor using parametrize_with_checks https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
def test_decisiontreeregressor():
    ...


# TODO: test DecisionTreeClassifier using parametrize_with_checks https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
def test_decisiontreeclassifier():
    ...
