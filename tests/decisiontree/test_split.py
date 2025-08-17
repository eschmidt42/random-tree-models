from typing import Generator

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from random_tree_models.decisiontree.node import Node
from random_tree_models.decisiontree.split import (
    BestSplit,
    check_if_split_sensible,
    find_best_split,
    get_column,
    get_thresholds_and_target_groups,
    select_arrays_for_child_node,
    select_thresholds,
)
from random_tree_models.decisiontree.split_objects import SplitScore
from random_tree_models.params import (
    ColumnSelectionMethod,
    ColumnSelectionParameters,
    ThresholdSelectionMethod,
    ThresholdSelectionParameters,
    TreeGrowthParameters,
)

from .conftest import (
    BOOL_OPTIONS_NONE_OKAY,
    FLOAT_OPTIONS_NONE_NOT_OKAY,
    INT_OPTIONS_NONE_NOT_OKAY,
)


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
        best = BestSplit(
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


class Test_select_thresholds:
    """
    bruteforce: returns all possible thresholds from the 2nd onward
    random:
    * returns a random subset of the thresholds if n_thresholds smaller than avaliable values
    * is reproducible with random_state
    quantile: returns num_quantile_steps thresholds which are ordered
    uniform: returns single value between min and max
    """

    def test_bruteforce(self):
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.bruteforce
        )
        feature_values = np.linspace(-1, 1, 100)
        rng = np.random.RandomState(42)

        # line to test
        thresholds = select_thresholds(feature_values, params, rng=rng)

        assert np.allclose(thresholds, feature_values[1:])

    def test_random_when_to_few_values(self):
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.random, n_thresholds=1000
        )
        feature_values = np.linspace(-1, 1, 100)
        rng = np.random.RandomState(42)

        # line to test
        thresholds = select_thresholds(feature_values, params, rng=rng)

        assert np.allclose(thresholds, feature_values[1:])

    def test_random_when_enough_values(self):
        n_thresholds = 10
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.random, n_thresholds=n_thresholds
        )
        feature_values = np.linspace(-1, 1, 100)
        rng = np.random.RandomState(42)

        # line to test
        thresholds0 = select_thresholds(feature_values, params, rng=rng)

        assert thresholds0.shape == (n_thresholds,)
        assert np.unique(thresholds0).shape == (n_thresholds,)

    def test_random_reproducible(self):
        n_thresholds = 10
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.random, n_thresholds=n_thresholds
        )
        feature_values = np.linspace(-1, 1, 100)

        # line to test
        rng = np.random.RandomState(42)
        thresholds0 = select_thresholds(feature_values, params, rng=rng)
        rng = np.random.RandomState(42)
        thresholds1 = select_thresholds(feature_values, params, rng=rng)

        assert np.allclose(thresholds0, thresholds1)

    def test_random_produces_changing_thresholds(self):
        n_thresholds = 10
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.random, n_thresholds=n_thresholds
        )
        feature_values = np.linspace(-1, 1, 100)
        rng = np.random.RandomState(42)

        # line to test
        thresholds0 = select_thresholds(feature_values, params, rng=rng)
        thresholds1 = select_thresholds(feature_values, params, rng=rng)

        assert not np.allclose(thresholds0, thresholds1)

    def test_quantile(self):
        n_thresholds = 10
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.quantile,
            n_thresholds=n_thresholds,
            quantile=0.1,
        )
        feature_values = np.linspace(-1, 1, 100)
        rng = np.random.RandomState(42)

        # line to test
        thresholds = select_thresholds(feature_values, params, rng=rng)

        assert thresholds.shape == (11,)
        assert (thresholds[1:] > thresholds[:-1]).all()

    def test_uniform(self):
        n_thresholds = 10
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.uniform, n_thresholds=n_thresholds
        )
        rng = np.random.RandomState(42)
        feature_values = rng.normal(loc=0, scale=1, size=100)

        # line to test
        thresholds = select_thresholds(feature_values, params, rng=rng)

        assert thresholds.shape == (1,)
        assert thresholds[0] >= feature_values.min()
        assert thresholds[0] <= feature_values.max()


class Test_get_thresholds_and_target_groups:
    """
    * preduces a generator
    * produces twice as many items to iterate in the case of missing values
    * each item contains the current threshold, the target groups and a boolean that indicates the default direction
    * the default direction is always None if there are no missing values and otherwise boolean
    """

    def test_produces_generator(self):
        feature_values = np.linspace(-1, 1, 10)
        threshold_params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.bruteforce
        )
        rng = np.random.RandomState(42)

        # line to test
        gen = get_thresholds_and_target_groups(
            feature_values, threshold_params, rng=rng
        )

        assert isinstance(gen, Generator)

    def test_finite_only_case(self):
        feature_values = np.linspace(-1, 1, 10)
        threshold_params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.bruteforce
        )
        rng = np.random.RandomState(42)

        # line to test
        thresholds_and_target_groups = get_thresholds_and_target_groups(
            feature_values, threshold_params, rng=rng
        )

        c = 0
        for (
            threshold,
            target_groups,
            default_direction_is_left,
        ) in thresholds_and_target_groups:
            assert isinstance(target_groups, np.ndarray)
            assert threshold in feature_values[1:]
            assert target_groups.dtype == bool
            assert default_direction_is_left is None
            c += 1

        assert c == len(feature_values[1:])

    def test_with_missing_case(self):
        feature_values = np.linspace(-1, 1, 10)
        feature_values[5] = np.nan
        threshold_params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.bruteforce
        )
        rng = np.random.RandomState(42)

        thresholds_and_target_groups = get_thresholds_and_target_groups(
            feature_values, threshold_params, rng=rng
        )

        # line to test
        c = 0
        for (
            threshold,
            target_groups,
            default_direction_is_left,
        ) in thresholds_and_target_groups:
            assert isinstance(target_groups, np.ndarray)
            assert threshold in feature_values[1:]
            assert target_groups.dtype == bool
            assert default_direction_is_left in [True, False]
            c += 1

        assert c == 2 * (len(feature_values[1:]) - 1)


class Test_get_column:
    """
    * method ascending just returns ascending integer list for columns
    * method random returns random integer list for columns
    * method largest_delta returns column indices with largest feature max-min differences first
    * if n_columns_to_try is given it is used to shorted the returned list
    """

    def test_ascending(self):
        n_columns = 10
        n_trials = None
        column_params = ColumnSelectionParameters(
            method=ColumnSelectionMethod.ascending, n_trials=n_trials
        )
        X = np.random.normal(size=(100, n_columns))
        rng = np.random.RandomState(42)

        # line to test
        columns = get_column(X, column_params, rng=rng)

        assert columns == list(range(n_columns))

    def test_ascending_first_n_trials_columns(self):
        n_columns = 10
        n_trials = 5
        column_params = ColumnSelectionParameters(
            method=ColumnSelectionMethod.ascending, n_trials=n_trials
        )
        X = np.random.normal(size=(100, n_columns))
        rng = np.random.RandomState(42)

        # line to test
        columns = get_column(X, column_params, rng=rng)

        assert columns == list(range(n_trials))

    def test_random(self):
        n_columns = 10
        n_trials = None
        column_params = ColumnSelectionParameters(
            method=ColumnSelectionMethod.random, n_trials=n_trials
        )
        X = np.random.normal(size=(100, n_columns))
        rng = np.random.RandomState(42)

        # line to test
        columns = get_column(X, column_params, rng=rng)

        assert not all([i0 < i1 for i0, i1 in zip(columns[:-1], columns[1:])])
        assert sorted(columns) == list(range(n_columns))

    def test_random_is_reproducible(self):
        n_columns = 10
        n_trials = None
        column_params = ColumnSelectionParameters(
            method=ColumnSelectionMethod.random, n_trials=n_trials
        )
        X = np.random.normal(size=(100, n_columns))

        # line to test
        rng = np.random.RandomState(42)
        columns0 = get_column(X, column_params, rng=rng)
        rng = np.random.RandomState(42)
        columns1 = get_column(X, column_params, rng=rng)

        assert columns0 == columns1

    def test_largest_delta(self):
        n_columns = 5
        n_trials = None
        column_params = ColumnSelectionParameters(
            method=ColumnSelectionMethod.largest_delta, n_trials=n_trials
        )
        rng = np.random.RandomState(42)
        X = np.array([[0, 0.001], [0, 0.01], [0, 0.1], [0, 1.0], [0, 10.0]]).T

        n_repetitions = 100
        all_columns = np.zeros((n_repetitions, n_columns), dtype=int)

        for i in range(n_repetitions):
            # line to test
            all_columns[i, :] = get_column(X, column_params, rng=rng)

        assert np.allclose(stats.mode(all_columns, axis=0).mode, [4, 3, 2, 1, 0])


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
        grow_params = TreeGrowthParameters(max_depth=2)
        try:
            # line to test
            best = find_best_split(
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
        grow_params = TreeGrowthParameters(max_depth=2)
        try:
            # line to test
            best = find_best_split(
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
        growth_params = TreeGrowthParameters(max_depth=2)
        try:
            # line to test
            best = find_best_split(
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
        growth_params = TreeGrowthParameters(max_depth=2)
        try:
            # line to test
            best = find_best_split(
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
    "best,parent_node,growth_params,is_no_sensible_split_exp",
    [
        # parent is None #1
        (
            BestSplit(score=-1.0, column=0, threshold=0.0, target_groups=np.array([])),
            None,
            TreeGrowthParameters(max_depth=2),
            False,
        ),
        # parent is None #2
        (
            BestSplit(score=-1.0, column=0, threshold=0.0, target_groups=np.array([])),
            Node(measure=SplitScore("bla")),
            TreeGrowthParameters(max_depth=2),
            False,
        ),
        # split is sufficient
        (
            BestSplit(
                score=-1.0,
                column=0,
                threshold=0.0,
                target_groups=np.array([False, True]),
            ),
            Node(measure=SplitScore("bla", value=-1.1)),
            TreeGrowthParameters(max_depth=2, min_improvement=0.01),
            False,
        ),
        # split is insufficient - because min gain not exceeded
        (
            BestSplit(
                score=-1.0,
                column=0,
                threshold=0.0,
                target_groups=np.array([False, True]),
            ),
            Node(measure=SplitScore("bla", value=-1.1)),
            TreeGrowthParameters(max_depth=2, min_improvement=0.2),
            True,
        ),
        # split is insufficient - because all items sorted left
        (
            BestSplit(
                score=-1.0,
                column=0,
                threshold=0.0,
                target_groups=np.array([True, True]),
            ),
            Node(measure=SplitScore("bla", value=-1.1)),
            TreeGrowthParameters(max_depth=2, min_improvement=0.0),
            True,
        ),
        # split is insufficient - because all items sorted right
        (
            BestSplit(
                score=-1.0,
                column=0,
                threshold=0.0,
                target_groups=np.array([False, False]),
            ),
            Node(measure=SplitScore("bla", value=-1.1)),
            TreeGrowthParameters(max_depth=2, min_improvement=0.0),
            True,
        ),
    ],
)
def test_check_if_split_sensible(
    best: BestSplit,
    parent_node: Node,
    growth_params: TreeGrowthParameters,
    is_no_sensible_split_exp: bool,
):
    # line to test
    is_not_sensible_split, gain = check_if_split_sensible(
        best, parent_node, growth_params
    )

    assert is_not_sensible_split == is_no_sensible_split_exp
    if parent_node is None or parent_node.measure.value is None:  # type: ignore
        assert gain is None


@pytest.mark.parametrize("go_left", [True, False])
def test_select_arrays_for_child_node(go_left: bool):
    best = BestSplit(
        score=1.0,
        column=0,
        threshold=2.0,
        target_groups=np.array([True, True, False]),
    )

    X = np.array([[1], [2], [3]])
    y = np.array([True, True, False])
    g = np.array([1, 2, 3])
    h = np.array([4, 5, 6])

    # line to test
    _X, _y, _g, _h = select_arrays_for_child_node(
        go_left=go_left,
        best=best,
        X=X,
        y=y,
        g=g,
        h=h,
    )
    assert _g is not None
    assert _h is not None
    if go_left:
        assert np.allclose(_X, X[:2])
        assert np.allclose(_y, y[:2])
        assert np.allclose(_g, g[:2])
        assert np.allclose(_h, h[:2])
    else:
        assert np.allclose(_X, X[2:])
        assert np.allclose(_y, y[2:])
        assert np.allclose(_g, g[2:])
        assert np.allclose(_h, h[2:])
