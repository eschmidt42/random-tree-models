from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.decisiontree as dtree
import random_tree_models.utils as utils

# first value in each tuple is the value to test and the second is the flag indicating if this should work
BOOL_OPTIONS_NONE_OKAY = [(False, True), (True, True), ("blub", False)]
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
    "int_val, float_val, node_val, str_val, bool_val",
    [
        (int_val, float_val, node_val, str_val, bool_val)
        for int_val in INT_OPTIONS_NONE_OKAY
        for float_val in FLOAT_OPTIONS_NONE_OKAY
        for node_val in NODE_OPTIONS_NONE_OKAY
        for str_val in STR_OPTIONS_NONE_OKAY
        for bool_val in BOOL_OPTIONS_NONE_OKAY
    ],
)
def test_Node(int_val, float_val, node_val, str_val, bool_val):
    array_column, array_column_okay = int_val
    threshold, threshold_okay = float_val
    prediction, prediction_okay = float_val
    left, left_okay = node_val
    right, right_okay = node_val
    n_obs, n_obs_okay = int_val
    reason, reason_okay = str_val
    default_is_left, default_is_left_okay = bool_val

    is_okay = all(
        [
            array_column_okay,
            threshold_okay,
            prediction_okay,
            left_okay,
            right_okay,
            n_obs_okay,
            reason_okay,
            default_is_left_okay,
        ]
    )
    measure = dtree.SplitScore(name="blub", value=1.0)
    try:
        # line to test
        node = dtree.Node(
            array_column=array_column,
            threshold=threshold,
            prediction=prediction,
            default_is_left=default_is_left,
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
            "default_is_left",
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
    "score,column,threshold,target_groups,default_is_left",
    [
        (score, column, threshold, target_groups, default_is_left)
        for score in FLOAT_OPTIONS_NONE_NOT_OKAY
        for column in INT_OPTIONS_NONE_NOT_OKAY
        for threshold in FLOAT_OPTIONS_NONE_NOT_OKAY
        for target_groups in [
            (np.array([1, 2, 3]), True),
            (np.array([]), True),
            (None, False),
        ]
        for default_is_left in BOOL_OPTIONS_NONE_OKAY
    ],
)
def test_BestSplit(score, column, threshold, target_groups, default_is_left):
    score, score_okay = score
    column, column_okay = column
    threshold, threshold_okay = threshold
    target_groups, target_groups_okay = target_groups
    default_is_left, default_is_left_okay = default_is_left

    is_okay = all(
        [
            score_okay,
            column_okay,
            threshold_okay,
            target_groups_okay,
            default_is_left_okay,
        ]
    )
    is_bad = not is_okay

    try:
        # line to test
        best = dtree.BestSplit(
            score=score,
            column=column,
            threshold=threshold,
            target_groups=target_groups,
            default_is_left=default_is_left,
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
        assert hasattr(best, "default_is_left")


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

    X_1D_missing = np.array(
        [
            [
                1,
            ],
            [
                np.nan,
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
    X_2D_missing = np.hstack((np.ones_like(X_1D_missing), X_1D_missing))

    y_1class = np.ones(X_1D.shape[0], dtype=bool)
    y_2class = np.array([False, False, True, True])
    y_3class = np.array([0, 0, 1, 2])

    y_1reg = np.ones(X_1D.shape[0])
    y_2reg = np.array([-1.0, -1.0, 1.0, 1.0])
    y_3reg = np.array([-1.0, -0.9, 1.0, 2.0])

    # xgboost - least squares
    g_1reg = np.array([0.0, 0.0, 0.0, 0.0])
    g_2reg = np.array([-1.0, -1.0, 1.0, 1.0])
    g_3reg = np.array([-1.275, -1.175, 0.725, 1.725])

    h_1reg = np.array([-1.0, -1.0, -1.0, -1.0])
    h_2reg = np.array([-1.0, -1.0, -1.0, -1.0])
    h_3reg = np.array([-1.0, -1.0, -1.0, -1.0])

    # xgboost - binomial log-likelihood
    g_2class = np.array([-1.0, -1.0, 1.0, 1.0])
    h_2class = np.array([-1.0, -1.0, -1.0, -1.0])

    @pytest.mark.parametrize(
        "y,ix,measure_name,g,h",
        [
            (y_1class, None, "gini", None, None),
            (y_2class, 2, "gini", None, None),
            (y_3class, 2, "gini", None, None),
            (y_1class, None, "entropy", None, None),
            (y_2class, 2, "entropy", None, None),
            (y_3class, 2, "entropy", None, None),
            (y_1reg, None, "variance", None, None),
            (y_2reg, 2, "variance", None, None),
            (y_3reg, 2, "variance", None, None),
            (y_1reg, None, "xgboost", g_1reg, h_1reg),
            (y_2reg, 2, "xgboost", g_2reg, h_2reg),
            (y_3reg, 2, "xgboost", g_3reg, h_3reg),
            # (y_1class, None, "xgboost", g_1class, h_1class), # currently not handled
            (y_2class, 2, "xgboost", g_2class, h_2class),
            # (y_3class, 2, "xgboost", g_3class, h_3class), # currently not handled
        ],
    )
    def test_1d(
        self,
        y: np.ndarray,
        ix: int,
        measure_name: str,
        g: np.ndarray,
        h: np.ndarray,
    ):
        is_homogenous = len(np.unique(y)) == 1
        grow_params = utils.TreeGrowthParameters()
        try:
            # line to test
            best = dtree.find_best_split(
                self.X_1D,
                y,
                measure_name=measure_name,
                g=g,
                h=h,
                growth_params=grow_params,
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
        "y,ix,measure_name,g,h",
        [
            (y_1class, None, "gini", None, None),
            (y_2class, 2, "gini", None, None),
            (y_3class, 2, "gini", None, None),
            (y_1class, None, "entropy", None, None),
            (y_2class, 2, "entropy", None, None),
            (y_3class, 2, "entropy", None, None),
            (y_1reg, None, "variance", None, None),
            (y_2reg, 2, "variance", None, None),
            (y_3reg, 2, "variance", None, None),
            (y_1reg, None, "xgboost", g_1reg, h_1reg),
            (y_2reg, 2, "xgboost", g_2reg, h_2reg),
            (y_3reg, 2, "xgboost", g_3reg, h_3reg),
            # (y_1class, None, "xgboost", g_1class, h_1class), # currently not handled
            (y_2class, 2, "xgboost", g_2class, h_2class),
            # (y_3class, 2, "xgboost", g_3class, h_3class), # currently not handled
        ],
    )
    def test_1d_missing(
        self,
        y: np.ndarray,
        ix: int,
        measure_name: str,
        g: np.ndarray,
        h: np.ndarray,
    ):
        is_homogenous = len(np.unique(y)) == 1
        grow_params = utils.TreeGrowthParameters()
        try:
            # line to test
            best = dtree.find_best_split(
                self.X_1D_missing,
                y,
                measure_name=measure_name,
                g=g,
                h=h,
                growth_params=grow_params,
            )
        except ValueError as ex:
            if is_homogenous:
                pytest.xfail("Splitting a homogneous y failed as expected")
            else:
                raise ex
        else:
            if is_homogenous:
                pytest.fail("Splitting a homogneous y passed unexpectedly")

            threshold_exp = float(self.X_1D_missing[ix, 0])
            assert best.threshold == threshold_exp

    @pytest.mark.parametrize(
        "y,ix,measure_name,g,h",
        [
            (y_1class, None, "gini", None, None),
            (y_2class, 2, "gini", None, None),
            (y_3class, 2, "gini", None, None),
            (y_1class, None, "entropy", None, None),
            (y_2class, 2, "entropy", None, None),
            (y_3class, 2, "entropy", None, None),
            (y_1reg, None, "variance", None, None),
            (y_2reg, 2, "variance", None, None),
            (y_3reg, 2, "variance", None, None),
            (y_1reg, None, "xgboost", g_1reg, h_1reg),
            (y_2reg, 2, "xgboost", g_2reg, h_2reg),
            (y_3reg, 2, "xgboost", g_3reg, h_3reg),
            # (y_1class, None, "xgboost", g_1class, h_1class), # currently not handled
            (y_2class, 2, "xgboost", g_2class, h_2class),
            # (y_3class, 2, "xgboost", g_3class, h_3class), # currently not handled
        ],
    )
    def test_2d(
        self,
        y: np.ndarray,
        ix: int,
        measure_name: str,
        g: np.ndarray,
        h: np.ndarray,
    ):
        is_homogenous = len(np.unique(y)) == 1
        growth_params = utils.TreeGrowthParameters()
        try:
            # line to test
            best = dtree.find_best_split(
                self.X_2D,
                y,
                measure_name,
                g=g,
                h=h,
                growth_params=growth_params,
            )
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

    @pytest.mark.parametrize(
        "y,ix,measure_name,g,h",
        [
            (y_1class, None, "gini", None, None),
            (y_2class, 2, "gini", None, None),
            (y_3class, 2, "gini", None, None),
            (y_1class, None, "entropy", None, None),
            (y_2class, 2, "entropy", None, None),
            (y_3class, 2, "entropy", None, None),
            (y_1reg, None, "variance", None, None),
            (y_2reg, 2, "variance", None, None),
            (y_3reg, 2, "variance", None, None),
            (y_1reg, None, "xgboost", g_1reg, h_1reg),
            (y_2reg, 2, "xgboost", g_2reg, h_2reg),
            (y_3reg, 2, "xgboost", g_3reg, h_3reg),
            # (y_1class, None, "xgboost", g_1class, h_1class), # currently not handled
            (y_2class, 2, "xgboost", g_2class, h_2class),
            # (y_3class, 2, "xgboost", g_3class, h_3class), # currently not handled
        ],
    )
    def test_2d_missing(
        self,
        y: np.ndarray,
        ix: int,
        measure_name: str,
        g: np.ndarray,
        h: np.ndarray,
    ):
        is_homogenous = len(np.unique(y)) == 1
        growth_params = utils.TreeGrowthParameters()
        try:
            # line to test
            best = dtree.find_best_split(
                self.X_2D_missing,
                y,
                measure_name,
                g=g,
                h=h,
                growth_params=growth_params,
            )
        except ValueError as ex:
            if is_homogenous:
                pytest.xfail("Splitting a homogneous y failed as expected")
            else:
                raise ex
        else:
            if is_homogenous:
                pytest.fail("Splitting a homogneous y passed unexpectedly")

            assert best.column == 1
            threshold_exp = float(self.X_2D_missing[ix, 1])
            assert best.threshold == threshold_exp


@pytest.mark.parametrize(
    "best,parent_node,growth_params,is_insufficient_exp",
    [
        # parent is None #1
        (
            dtree.BestSplit(
                score=-1.0, column=0, threshold=0.0, target_groups=np.array([])
            ),
            None,
            utils.TreeGrowthParameters(),
            False,
        ),
        # parent is None #2
        (
            dtree.BestSplit(
                score=-1.0, column=0, threshold=0.0, target_groups=np.array([])
            ),
            dtree.Node(measure=dtree.SplitScore("bla")),
            utils.TreeGrowthParameters(),
            False,
        ),
        # split is sufficient
        (
            dtree.BestSplit(
                score=-1.0, column=0, threshold=0.0, target_groups=np.array([])
            ),
            dtree.Node(measure=dtree.SplitScore("bla", value=-1.1)),
            utils.TreeGrowthParameters(min_improvement=0.01),
            False,
        ),
        # split is insufficient
        (
            dtree.BestSplit(
                score=-1.0, column=0, threshold=0.0, target_groups=np.array([])
            ),
            dtree.Node(measure=dtree.SplitScore("bla", value=-1.1)),
            utils.TreeGrowthParameters(min_improvement=0.2),
            True,
        ),
    ],
)
def test_check_if_gain_insufficient(
    best: dtree.BestSplit,
    parent_node: dtree.Node,
    growth_params: utils.TreeGrowthParameters,
    is_insufficient_exp: bool,
):
    # line to test
    is_insufficient_gain, gain = dtree.check_if_gain_insufficient(
        best, parent_node, growth_params
    )

    assert is_insufficient_gain == is_insufficient_exp
    if parent_node is None or parent_node.measure.value is None:
        assert gain is None


class Test_grow_tree:
    X = np.array([[1], [2], [3]])
    y = np.array([True, True, False])
    target_groups = np.array([True, True, False])
    measure_name = "gini"
    depth_dummy = 0

    def test_baselevel(self):
        # test returned leaf node
        growth_params = utils.TreeGrowthParameters()
        parent_node = None
        is_baselevel = True
        reason = "very custom leaf node comment"
        with patch(
            "random_tree_models.decisiontree.check_is_baselevel",
            return_value=[is_baselevel, reason],
        ) as mock_check_is_baselevel:
            # line to test
            leaf_node = dtree.grow_tree(
                self.X,
                self.y,
                self.measure_name,
                parent_node=parent_node,
                depth=self.depth_dummy,
                growth_params=growth_params,
            )

            mock_check_is_baselevel.assert_called_once()
            assert leaf_node.is_leaf == True
            assert leaf_node.reason == reason

    def test_split_improvement_insufficient(self):
        # test split improvement below minimum
        growth_params = utils.TreeGrowthParameters(min_improvement=0.2)
        parent_score = -1.0
        new_score = -0.9
        best = dtree.BestSplit(
            score=new_score,
            column=0,
            threshold=3.0,
            target_groups=self.target_groups,
        )
        measure = dtree.SplitScore(self.measure_name, parent_score)
        parent_node = dtree.Node(
            array_column=0,
            threshold=1.0,
            prediction=0.9,
            left=None,
            right=None,
            measure=measure,
            n_obs=3,
            reason="",
        )
        is_baselevel = False
        leaf_reason = "very custom leaf node comment"
        gain = new_score - parent_score
        split_reason = f"gain due split ({gain=}) lower than {growth_params.min_improvement=}"
        with (
            patch(
                "random_tree_models.decisiontree.check_is_baselevel",
                return_value=[is_baselevel, leaf_reason],
            ) as mock_check_is_baselevel,
            patch(
                "random_tree_models.decisiontree.find_best_split",
                return_value=best,
            ) as mock_find_best_split,
        ):
            # line to test
            node = dtree.grow_tree(
                self.X,
                self.y,
                self.measure_name,
                parent_node=parent_node,
                depth=self.depth_dummy,
                growth_params=growth_params,
            )

            mock_check_is_baselevel.assert_called_once()
            mock_find_best_split.assert_called_once()
            assert node.reason == split_reason
            assert node.prediction == np.mean(self.y)
            assert node.n_obs == len(self.y)

    def test_split_improvement_sufficient(self):
        # test split improvement above minumum, leading to two leaf nodes
        growth_params = utils.TreeGrowthParameters(min_improvement=0.0)
        parent_score = -1.0
        new_score = -0.9
        best = dtree.BestSplit(
            score=new_score,
            column=0,
            threshold=3.0,
            target_groups=self.target_groups,
        )
        measure = dtree.SplitScore(self.measure_name, parent_score)
        parent_node = dtree.Node(
            array_column=0,
            threshold=1.0,
            prediction=0.9,
            left=None,
            right=None,
            measure=measure,
            n_obs=3,
            reason="",
        )

        leaf_reason = "very custom leaf node comment"

        with (
            patch(
                "random_tree_models.decisiontree.check_is_baselevel",
                side_effect=[
                    (False, "bla"),
                    (True, leaf_reason),
                    (True, leaf_reason),
                ],
            ) as mock_check_is_baselevel,
            patch(
                "random_tree_models.decisiontree.find_best_split",
                side_effect=[best],
            ) as mock_find_best_split,
        ):
            # line to test
            tree = dtree.grow_tree(
                self.X,
                self.y,
                self.measure_name,
                parent_node=parent_node,
                depth=self.depth_dummy,
                growth_params=growth_params,
            )

            assert mock_check_is_baselevel.call_count == 3
            assert mock_find_best_split.call_count == 1

            # parent
            assert tree.reason == ""
            assert tree.prediction == np.mean(self.y)
            assert tree.n_obs == len(self.y)
            assert tree.is_leaf == False

            # left leaf
            assert tree.left.reason == leaf_reason
            assert tree.left.prediction == 1.0
            assert tree.left.n_obs == 2
            assert tree.left.is_leaf == True

            # right leaf
            assert tree.right.reason == leaf_reason
            assert tree.right.prediction == 0.0
            assert tree.right.n_obs == 1
            assert tree.right.is_leaf == True


@pytest.mark.parametrize(
    "x,exp",
    [
        (np.array([-1, -1]), 0.0),
        (np.array([1, -1]), 1.0),
        (np.array([1, 1]), 2.0),
        (np.array([-1, 1]), 3.0),
    ],
)
def test_find_leaf_node(x: np.ndarray, exp: float):
    tree = dtree.Node(
        array_column=0,
        threshold=0.0,
        left=dtree.Node(
            array_column=1,
            threshold=0.0,
            left=dtree.Node(prediction=0.0),
            right=dtree.Node(prediction=3.0),
        ),
        right=dtree.Node(
            array_column=1,
            threshold=0.0,
            left=dtree.Node(prediction=1.0),
            right=dtree.Node(prediction=2.0),
        ),
    )
    # line to test
    leaf = dtree.find_leaf_node(tree, x)

    assert leaf.prediction == exp


def test_predict_with_tree():
    X = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    tree = dtree.Node(
        array_column=0,
        threshold=0.0,
        left=dtree.Node(
            array_column=1,
            threshold=0.0,
            left=dtree.Node(prediction=0.0),
            right=dtree.Node(prediction=3.0),
        ),
        right=dtree.Node(
            array_column=1,
            threshold=0.0,
            left=dtree.Node(prediction=1.0),
            right=dtree.Node(prediction=2.0),
        ),
    )

    # line to test
    predictions = dtree.predict_with_tree(tree, X)

    assert np.allclose(predictions, np.arange(0, 4, 1))


class TestDecisionTreeTemplate:
    model = dtree.DecisionTreeTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "tree_")

    def test_growth_params_(self):
        assert not hasattr(self.model, "growth_params_")

        self.model._organize_growth_parameters()
        assert isinstance(self.model.growth_params_, utils.TreeGrowthParameters)

    def test_fit(self):
        try:
            self.model.fit(None, None)
        except NotImplementedError as ex:
            pytest.xfail("DecisionTreeTemplate.fit expectedly refused call")

    def test_predict(self):
        try:
            self.model.predict(None)
        except NotImplementedError as ex:
            pytest.xfail("DecisionTreeTemplate.predict expectedly refused call")


class TestDecisionTreeRegressor:
    model = dtree.DecisionTreeRegressor()

    X = np.array(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])

    def test_fit(self):
        model = dtree.DecisionTreeRegressor()
        model.fit(self.X, self.y)
        assert isinstance(model.tree_, dtree.Node)

    def test_predict(self):
        model = dtree.DecisionTreeRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestDecisionTreeClassifier:
    model = dtree.DecisionTreeClassifier()

    X = np.array(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    y = np.array([False, False, True, True])

    def test_classes_(self):
        assert not hasattr(self.model, "classes_")

    def test_fit(self):
        model = dtree.DecisionTreeClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert isinstance(model.tree_, dtree.Node)

    def test_predict(self):
        model = dtree.DecisionTreeClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@pytest.mark.slow
@parametrize_with_checks(
    [dtree.DecisionTreeRegressor(), dtree.DecisionTreeClassifier()]
)
def test_dtree_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)
