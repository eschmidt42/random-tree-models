import typing as T

import numpy as np

import random_tree_models.scoring as scoring
from random_tree_models.models.decisiontree.node import Node
from random_tree_models.models.decisiontree.split_objects import BestSplit
from random_tree_models.params import (
    ColumnSelectionMethod,
    ColumnSelectionParameters,
    MetricNames,
    ThresholdSelectionMethod,
    ThresholdSelectionParameters,
    TreeGrowthParameters,
)


def select_thresholds(
    feature_values: np.ndarray,
    threshold_params: ThresholdSelectionParameters,
    rng: np.random.RandomState,
) -> np.ndarray:
    "Selects thresholds to use for splitting"

    method = threshold_params.method
    n_thresholds = threshold_params.n_thresholds
    num_quantile_steps = threshold_params.num_quantile_steps

    if method == ThresholdSelectionMethod.bruteforce:
        return feature_values[1:]
    elif method == ThresholdSelectionMethod.random:
        if len(feature_values) - 1 <= n_thresholds:
            return feature_values[1:]
        else:
            return rng.choice(
                feature_values[1:],
                size=(n_thresholds,),
                replace=False,
            )
    elif method == ThresholdSelectionMethod.quantile:
        qs = np.linspace(0, 1, num_quantile_steps)
        return np.quantile(feature_values[1:], qs)
    elif method == ThresholdSelectionMethod.uniform:
        x = np.linspace(
            feature_values.min(),
            feature_values.max(),
            n_thresholds + 2,
        )
        return rng.choice(x[1:], size=[1])
    else:
        raise NotImplementedError(f"Unknown threshold selection method: {method}")


def get_thresholds_and_target_groups(
    feature_values: np.ndarray,
    threshold_params: ThresholdSelectionParameters,
    rng: np.random.RandomState,
) -> T.Generator[T.Tuple[np.ndarray, np.ndarray, bool | None], None, None]:
    "Creates a generator for split finding, returning the used threshold, the target groups and a bool indicating if the default direction is left"
    is_missing = np.isnan(feature_values)
    is_finite = np.logical_not(is_missing)
    all_finite = is_finite.all()

    if all_finite:
        default_direction_is_left = None
        thresholds = select_thresholds(feature_values, threshold_params, rng)

        for threshold in thresholds:
            target_groups = feature_values < threshold
            yield (threshold, target_groups, default_direction_is_left)
    else:
        finite_feature_values = feature_values[is_finite]
        thresholds = select_thresholds(finite_feature_values, threshold_params, rng)

        for threshold in thresholds:
            # default direction left - feature value <= threshold or missing  (i.e. missing are included left of the threshold)
            target_groups = np.logical_or(feature_values < threshold, is_missing)
            yield (threshold, target_groups, True)

            # default direction right - feature value <= threshold and finite (i.e. missing are included right of the threshold)
            target_groups = np.logical_and(feature_values < threshold, is_finite)
            yield (threshold, target_groups, False)


def get_column(
    X: np.ndarray,
    column_params: ColumnSelectionParameters,
    rng: np.random.RandomState,
) -> list[int]:
    # select column order to split on
    method = column_params.method
    n_columns_to_try = column_params.n_trials

    columns = list(range(X.shape[1]))
    if method == ColumnSelectionMethod.ascending:
        pass
    elif method == ColumnSelectionMethod.random:
        columns = np.array(columns)
        rng.shuffle(columns)
        columns = columns.tolist()
    elif method == ColumnSelectionMethod.largest_delta:
        deltas = X.max(axis=0) - X.min(axis=0)
        weights = deltas / deltas.sum()
        columns = np.array(columns)
        columns = rng.choice(columns, p=weights, size=len(columns), replace=False)
        columns = columns.tolist()
    else:
        raise NotImplementedError(
            f"Unknown column selection method: {column_params.method}"
        )
    if n_columns_to_try is not None:
        columns = columns[:n_columns_to_try]

    return columns


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: str,
    yhat: np.ndarray | None = None,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    growth_params: TreeGrowthParameters | None = None,  # TODO: make required
    rng: np.random.RandomState = np.random.RandomState(42),
) -> BestSplit:
    """Find the best split, detecting the "default direction" with missing data."""

    if len(np.unique(y)) == 1:
        raise ValueError(
            f"Tried to find a split for homogenous y: {y[:3]} ... {y[-3:]}"
        )

    best = None  # this will be an BestSplit instance

    if growth_params is None:
        raise ValueError(f"{growth_params=} but is not allowed to be None")

    for array_column in get_column(X, growth_params.column_params, rng):
        feature_values = X[:, array_column]

        for (
            threshold,
            target_groups,
            default_is_left,
        ) in get_thresholds_and_target_groups(
            feature_values, growth_params.threshold_params, rng
        ):
            split_score = scoring.calc_split_score(
                MetricNames(measure_name),
                y,
                target_groups,
                yhat=yhat,
                g=g,
                h=h,
                growth_params=growth_params,
            )

            if best is None or split_score > best.score:
                best = BestSplit(
                    score=float(split_score),
                    column=int(array_column),
                    threshold=float(threshold),
                    target_groups=target_groups,
                    default_is_left=default_is_left,
                )

    if best is None:
        raise ValueError(f"Something went wrong {best=} cannot be None.")
    return best


def check_if_split_sensible(
    best: BestSplit,
    parent_node: Node | None,
    growth_params: TreeGrowthParameters,
) -> tuple[bool, float | None]:
    "Verifies if split is sensible, considering score gain and left/right group sizes"
    parent_is_none = parent_node is None
    if parent_is_none:
        return False, None

    measure_is_none = parent_node.measure is None
    if measure_is_none:
        return False, None

    value_is_none = parent_node.measure.value is None  # type: ignore
    if value_is_none:
        return False, None

    # score gain
    gain = best.score - parent_node.measure.value  # type: ignore
    is_insufficient_gain = gain < growth_params.min_improvement

    # left/right group assignment
    all_on_one_side = bool(best.target_groups.all())
    all_on_other_side = bool(np.logical_not(best.target_groups).all())
    is_all_onesided = all_on_one_side or all_on_other_side

    is_not_sensible = is_all_onesided or is_insufficient_gain

    return is_not_sensible, gain


def select_arrays_for_child_node(
    go_left: bool,
    best: BestSplit,
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mask = best.target_groups == go_left
    _X = X[mask, :]
    _y = y[mask]
    _g = g[mask] if g is not None else None
    _h = h[mask] if h is not None else None
    return _X, _y, _g, _h
