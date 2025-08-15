import typing as T
import uuid

import numpy as np
import pandas as pd
import sklearn.base as base
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from pydantic.dataclasses import dataclass
from rich import print as rprint
from rich.tree import Tree
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # type: ignore
)

import random_tree_models.leafweights as leafweights
import random_tree_models.scoring as scoring
import random_tree_models.utils as utils

logger = utils.logger


@dataclass(validate_on_init=True)
class SplitScore:
    name: StrictStr  # name of the score used
    value: StrictFloat | None = None  # optimization value gini etc


@dataclass
class Node:
    """Decision node in a decision tree"""

    # Stuff for making a decision
    array_column: StrictInt | None = None  # index of the column to use
    threshold: float | None = None  # threshold for decision
    prediction: float | None = None  # value to use for predictions
    default_is_left: bool | None = None  # default direction is x is nan

    # decendants
    right: "Node | None" = None  # right decendany of type Node
    left: "Node | None" = None  # left decendant of type Node

    # misc info
    measure: SplitScore | None = None

    n_obs: StrictInt | None = None  # number of observations in node
    reason: StrictStr | None = None  # place for some comment

    depth: StrictInt | None = None  # depth of the node

    def __post_init__(self):
        # unique identifier of the node
        self.node_id = uuid.uuid4()

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def check_is_baselevel(y: np.ndarray, depth: int, max_depth: int) -> T.Tuple[bool, str]:
    """Verifies if the tree traversal reached the baselevel / a leaf
    * group homogeneous / cannot sensibly be splitted further
    * no data in the group
    * max depth reached
    """
    if max_depth is not None and depth >= max_depth:
        return (True, "max depth reached")
    elif len(np.unique(y)) == 1:
        return (True, "homogenous group")
    elif len(y) <= 1:
        return (True, "<= 1 data point in group")
    else:
        return (False, "")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BestSplit:
    score: StrictFloat
    column: StrictInt
    threshold: StrictFloat
    target_groups: np.ndarray = Field(default_factory=lambda: np.zeros(10))
    default_is_left: StrictBool | None = None


def select_thresholds(
    feature_values: np.ndarray,
    threshold_params: utils.ThresholdSelectionParameters,
    rng: np.random.RandomState,
) -> np.ndarray:
    "Selects thresholds to use for splitting"

    method = threshold_params.method
    n_thresholds = threshold_params.n_thresholds
    num_quantile_steps = threshold_params.num_quantile_steps

    if method == utils.ThresholdSelectionMethod.bruteforce:
        return feature_values[1:]
    elif method == utils.ThresholdSelectionMethod.random:
        if len(feature_values) - 1 <= n_thresholds:
            return feature_values[1:]
        else:
            return rng.choice(
                feature_values[1:],
                size=(n_thresholds,),
                replace=False,
            )
    elif method == utils.ThresholdSelectionMethod.quantile:
        qs = np.linspace(0, 1, num_quantile_steps)
        return np.quantile(feature_values[1:], qs)
    elif method == utils.ThresholdSelectionMethod.uniform:
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
    threshold_params: utils.ThresholdSelectionParameters,
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
    column_params: utils.ColumnSelectionParameters,
    rng: np.random.RandomState,
) -> list[int]:
    # select column order to split on
    method = column_params.method
    n_columns_to_try = column_params.n_trials

    columns = list(range(X.shape[1]))
    if method == utils.ColumnSelectionMethod.ascending:
        pass
    elif method == utils.ColumnSelectionMethod.random:
        columns = np.array(columns)
        rng.shuffle(columns)
        columns = columns.tolist()
    elif method == utils.ColumnSelectionMethod.largest_delta:
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
    growth_params: utils.TreeGrowthParameters | None = None,  # TODO: make required
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
            split_score = scoring.SplitScoreMetrics[measure_name](
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
    parent_node: Node,
    growth_params: utils.TreeGrowthParameters,
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


def calc_leaf_weight_and_split_score(
    y: np.ndarray,
    measure_name: str,
    growth_params: utils.TreeGrowthParameters,
    g: np.ndarray,
    h: np.ndarray,
) -> tuple[float, float]:
    leaf_weight = leafweights.calc_leaf_weight(y, measure_name, growth_params, g=g, h=h)

    yhat = leaf_weight * np.ones_like(y)
    score = scoring.SplitScoreMetrics[measure_name](
        y,
        np.ones_like(y, dtype=bool),
        yhat=yhat,
        g=g,
        h=h,
        growth_params=growth_params,
    )

    return leaf_weight, score


def select_arrays_for_child_node(
    go_left: bool,
    best: BestSplit,
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mask = best.target_groups == go_left
    _X = X[mask, :]
    _y = y[mask]
    _g = g[mask] if g is not None else None
    _h = h[mask] if h is not None else None
    return _X, _y, _g, _h


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: str,
    growth_params: utils.TreeGrowthParameters,
    parent_node: Node | None = None,
    depth: int = 0,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    random_state: int = 42,
    **kwargs,
) -> Node:
    """Implementation of the Classification And Regression Tree (CART) algorithm

    Args:
        X (np.ndarray): Input feature values to do thresholding on.
        y (np.ndarray): Target values.
        measure_name (str): Values indicating which functions in scoring.SplitScoreMetrics and leafweights.LeafWeightSchemes to call.
        parent_node (Node, optional): Parent node in tree. Defaults to None.
        depth (int, optional): Current tree depth. Defaults to 0.
        growth_params (utils.TreeGrowthParameters, optional): Parameters controlling tree growth. Defaults to None.
        g (np.ndarray, optional): Boosting and loss specific precomputed 1st order derivative dloss/dyhat. Defaults to None.
        h (np.ndarray, optional): Boosting and loss specific precomputed 2nd order derivative d^2loss/dyhat^2. Defaults to None.

    Raises:
        ValueError: Fails if parent node passes an empty y array.

    Returns:
        Node: Tree node with leaf weight, node score and potential child nodes.

    Note:
    Currently measure_name controls how the split score and the leaf weights are computed.

    But only the decision tree algorithm directly uses y for that and can predict y using the leaf weight values directly.

    For the boosting algorithms g and h are used to compute split score and leaf weights. Their leaf weights
    sometimes also need post-processing, e.g. for binary classification. Computation of g and h and post-processing is not
    done here but in the respective class implementations of the algorithms.
    """

    n_obs = len(y)
    if n_obs == 0:
        raise ValueError(
            f"Something went wrong. {parent_node=} handed down an empty set of data points."
        )

    is_baselevel, reason = check_is_baselevel(
        y, depth, max_depth=growth_params.max_depth
    )

    # compute leaf weight (for prediction) and node score (for split gain check)
    leaf_weight, score = calc_leaf_weight_and_split_score(
        y, measure_name, growth_params, g, h
    )

    if is_baselevel:  # end of the line buddy
        return Node(
            prediction=leaf_weight,
            measure=SplitScore(measure_name, value=score),
            n_obs=n_obs,
            reason=reason,
            depth=depth,
        )

    # find best split
    rng = np.random.RandomState(random_state)

    best = find_best_split(
        X, y, measure_name, g=g, h=h, growth_params=growth_params, rng=rng
    )

    # check if improvement due to split is below minimum requirement
    is_not_sensible_split, gain = check_if_split_sensible(
        best, parent_node, growth_params
    )

    if is_not_sensible_split:
        reason = f"gain due split ({gain=}) lower than {growth_params.min_improvement=} or all data points assigned to one side (is left {best.target_groups.mean()=:.2%})"
        leaf_node = Node(
            prediction=leaf_weight,
            measure=SplitScore(measure_name, value=score),
            n_obs=n_obs,
            reason=reason,
            depth=depth,
        )
        return leaf_node

    # create new parent node for subsequent child nodes
    new_node = Node(
        array_column=best.column,
        threshold=best.threshold,
        prediction=leaf_weight,
        default_is_left=best.default_is_left,
        measure=SplitScore(measure_name, best.score),
        n_obs=n_obs,
        reason="",
        depth=depth,
    )
    random_state_left, random_state_right = rng.randint(0, 2**32, size=2)

    # descend left
    _X, _y, _g, _h = select_arrays_for_child_node(True, best, X, y, g, h)
    new_node.left = grow_tree(
        _X,
        _y,
        measure_name=measure_name,
        growth_params=growth_params,
        parent_node=new_node,
        depth=depth + 1,
        g=_g,
        h=_h,
        random_state=random_state_left,
    )

    # descend right
    _X, _y, _g, _h = select_arrays_for_child_node(False, best, X, y, g, h)
    new_node.right = grow_tree(
        _X,
        _y,
        measure_name=measure_name,
        growth_params=growth_params,
        parent_node=new_node,
        depth=depth + 1,
        g=_g,
        h=_h,
        random_state=random_state_right,
    )

    return new_node


def find_leaf_node(node: Node, x: np.ndarray) -> Node:
    "Traverses tree to find the leaf corresponding to x"

    if node.is_leaf:
        return node

    is_missing = np.isnan(x[node.array_column])
    if is_missing:
        go_left = node.default_is_left
        if go_left is None:
            raise ValueError(
                f"{x[node.array_column]=} is missing but was not observed as a feature that can be missing during training."
            )
    else:
        go_left = x[node.array_column] < node.threshold
    if go_left:
        node = find_leaf_node(node.left, x)
    else:
        node = find_leaf_node(node.right, x)
    return node


def predict_with_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    "Traverse a previously built tree to make one prediction per row in X"
    if not isinstance(tree, Node):
        raise ValueError(
            f"Passed `tree` needs to be an instantiation of Node, got {tree=}"
        )
    n_obs = len(X)
    predictions = [None for _ in range(n_obs)]

    for i in range(n_obs):
        leaf_node = find_leaf_node(tree, X[i, :])

        predictions[i] = leaf_node.prediction

    predictions = np.array(predictions)
    return predictions


class DecisionTreeTemplate(base.BaseEstimator):
    """Template for DecisionTree classes

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: str = None,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: utils.ThresholdSelectionMethod = "bruteforce",
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: utils.ColumnSelectionMethod = "ascending",
        n_columns_to_try: int = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
        # **kwargs
    ) -> None:
        self.max_depth = max_depth
        self.measure_name = measure_name
        self.min_improvement = min_improvement
        self.lam = lam
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.random_state = random_state
        self.threshold_method = threshold_method
        self.threshold_quantile = threshold_quantile
        self.n_thresholds = n_thresholds
        self.column_method = column_method
        self.n_columns_to_try = n_columns_to_try
        self.ensure_all_finite = ensure_all_finite

    def _organize_growth_parameters(self):
        self.growth_params_ = utils.TreeGrowthParameters(
            max_depth=self.max_depth,
            min_improvement=self.min_improvement,
            lam=-abs(self.lam),
            frac_subsamples=float(self.frac_subsamples),
            frac_features=float(self.frac_features),
            random_state=int(self.random_state),
            threshold_params=utils.ThresholdSelectionParameters(
                method=self.threshold_method,
                quantile=self.threshold_quantile,
                n_thresholds=self.n_thresholds,
                random_state=int(self.random_state),
            ),
            column_params=utils.ColumnSelectionParameters(
                method=self.column_method,
                n_trials=self.n_columns_to_try,
            ),
        )

    def _select_samples_and_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Sub-samples rows and columns from X and y"
        if not hasattr(self, "growth_params_"):
            raise ValueError(f"Try calling `fit` first.")

        ix = np.arange(len(X))
        rng = np.random.RandomState(self.growth_params_.random_state)
        if self.growth_params_.frac_subsamples < 1.0:
            n_samples = int(self.growth_params_.frac_subsamples * len(X))
            ix_samples = rng.choice(ix, size=n_samples, replace=False)
        else:
            ix_samples = ix

        if self.frac_features < 1.0:
            n_columns = int(X.shape[1] * self.frac_features)
            ix_features = rng.choice(
                np.arange(X.shape[1]),
                size=n_columns,
                replace=False,
            )
        else:
            ix_features = np.arange(X.shape[1])

        _X = X[ix_samples, :]
        _X = _X[:, ix_features]

        _y = y[ix_samples]
        return _X, _y, ix_features

    def _select_features(self, X: np.ndarray, ix_features: np.ndarray) -> np.ndarray:
        return X[:, ix_features]

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeTemplate":
        raise NotImplementedError()

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class DecisionTreeRegressor(base.RegressorMixin, DecisionTreeTemplate):
    """DecisionTreeRegressor

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: str = "variance",
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: utils.ThresholdSelectionMethod = "bruteforce",
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: utils.ColumnSelectionMethod = "ascending",
        n_columns_to_try: int = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
        # **kwargs,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            max_depth=max_depth,
            min_improvement=min_improvement,
            lam=lam,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
            n_thresholds=n_thresholds,
            column_method=column_method,
            n_columns_to_try=n_columns_to_try,
            random_state=random_state,
            ensure_all_finite=ensure_all_finite,
        )

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "DecisionTreeRegressor":
        self._organize_growth_parameters()
        # X, y = check_X_y(X, y, force_all_finite=self.ensure_all_finite)
        X, y = validate_data(self, X, y)
        # self.n_features_in_ = X.shape[1]

        _X, _y, self.ix_features_ = self._select_samples_and_features(X, y)

        self.tree_ = grow_tree(
            _X,
            _y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
            random_state=self.random_state,
            **kwargs,
        )

        return self

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ("tree_", "growth_params_"))

        X = validate_data(self, X, reset=False)
        # X = check_array(X, force_all_finite=self.ensure_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        _X = self._select_features(X, self.ix_features_)

        y = predict_with_tree(self.tree_, _X)

        return y


class DecisionTreeClassifier(base.ClassifierMixin, DecisionTreeTemplate):
    """DecisionTreeClassifier

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: str = "gini",
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: utils.ThresholdSelectionMethod = "bruteforce",
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: utils.ColumnSelectionMethod = "ascending",
        n_columns_to_try: int = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            max_depth=max_depth,
            min_improvement=min_improvement,
            lam=lam,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
            n_thresholds=n_thresholds,
            column_method=column_method,
            n_columns_to_try=n_columns_to_try,
            random_state=random_state,
        )
        self.ensure_all_finite = ensure_all_finite

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def __sklearn_tags__(self):
        # https://scikit-learn.org/stable/developers/develop.html
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeClassifier":
        X, y = validate_data(self, X, y)
        # X, y = check_X_y(X, y, ensure_all_finite=self.ensure_all_finite)
        check_classification_targets(y)

        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )

        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self._organize_growth_parameters()

        # self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)

        _X, _y, self.ix_features_ = self._select_samples_and_features(X, y)

        self.tree_ = grow_tree(
            _X,
            _y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
            random_state=self.random_state,
        )

        return self

    def predict_proba(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ("tree_", "classes_", "growth_params_"))
        X = validate_data(self, X, reset=False)
        # X = check_array(X, ensure_all_finite=self.ensure_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        _X = self._select_features(X, self.ix_features_)

        proba = predict_with_tree(self.tree_, _X)
        proba = np.array([1 - proba, proba]).T
        return proba

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y


def walk_tree(
    decision_tree: Node, tree: Tree, parent: Node = None, is_left: bool = None
):
    arrow = (
        ""
        if parent is None
        else f"[magenta](< {parent.threshold:.3f})[/magenta]"
        if is_left
        else f"[magenta](>= {parent.threshold:.3f})[/magenta]"
    )

    if decision_tree.is_leaf:  # base cases
        branch = tree.add(
            f"{arrow} 🍁 # obs: [cyan]{decision_tree.n_obs}[/cyan], value: [green]{decision_tree.prediction:.3f}[/green], leaf reason '{decision_tree.reason}'"
        )
        return None
    else:
        branch = tree.add(
            f"{arrow} col idx: {decision_tree.array_column}, threshold: [magenta]{decision_tree.threshold:.3f}[/magenta]"
        )

        if decision_tree.left is not None:  # go left
            walk_tree(decision_tree.left, branch, decision_tree, True)

        if decision_tree.right is not None:  # go right
            walk_tree(decision_tree.right, branch, decision_tree, False)


def show_tree(decision_tree: DecisionTreeTemplate):
    tree = Tree(f"Represenation of 🌲 ({decision_tree})")
    walk_tree(decision_tree.tree_, tree)
    rprint(tree)
