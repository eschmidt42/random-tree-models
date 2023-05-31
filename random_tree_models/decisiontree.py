import typing as T
import uuid
from enum import Enum
from functools import partial

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
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass(validate_on_init=True)
class SplitScore:
    name: StrictStr  # name of the score used
    value: StrictFloat = None  # optimization value gini etc


@dataclass
class Node:
    """Decision node in a decision tree"""

    # Stuff for making a decision
    array_column: StrictInt = None  # index of the column to use
    threshold: float = None  # threshold for decision
    prediction: float = None  # value to use for predictions

    # decendants
    left: "Node" = None  # left decendant of type Node
    right: "Node" = None  # right decendany of type Node

    # misc info
    measure: SplitScore = None

    n_obs: StrictInt = None  # number of observations in node
    reason: StrictStr = None  # place for some comment

    def __post_init__(self):
        # unique identifier of the node
        self.node_id = uuid.uuid4()

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def check_is_baselevel(
    y: np.ndarray, node: Node, depth: int, max_depth: int = None
) -> T.Tuple[bool, str]:
    "Verifies if the tree traversal reached the baselevel / a leaf"
    if depth >= max_depth:
        return (True, "max depth reached")
    elif len(np.unique(y)) == 1:
        return (True, "homogenous group")
    elif len(y) == 0:
        return (True, "no data in group")
    else:
        return (False, "")


def check_y_and_target_groups(y: np.ndarray, target_groups: np.ndarray = None):
    n = len(y)
    if n == 0:
        raise ValueError(f"{n=}, expected at least one target value")

    if target_groups is not None and len(target_groups) != n:
        raise ValueError(f"{y.shape=} != {target_groups.shape=}")


def calc_variance(y: np.ndarray, target_groups: np.ndarray) -> float:
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


def calc_entropy(y: np.ndarray, target_groups: np.ndarray) -> float:
    """Calculates the entropy of a split"""

    check_y_and_target_groups(y, target_groups=target_groups)

    w_left = target_groups.sum() / len(target_groups)
    w_right = 1.0 - w_left

    h_left = entropy(y[target_groups]) if w_left > 0 else 0
    h_right = entropy(y[~target_groups]) if w_right > 0 else 0

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


def calc_gini_impurity(y: np.ndarray, target_groups: np.ndarray) -> float:
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


class SplitScoreMetrics(Enum):
    # https://stackoverflow.com/questions/40338652/how-to-define-enum-values-that-are-functions
    variance = partial(calc_variance)
    entropy = partial(calc_entropy)
    gini = partial(calc_gini_impurity)

    def __call__(self, y: np.ndarray, target_groups: np.ndarray) -> float:
        return self.value(y, target_groups)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BestSplit:
    score: StrictFloat
    column: StrictInt
    threshold: StrictFloat
    target_groups: np.ndarray = Field(default_factory=lambda: np.zeros(10))


def find_best_split(
    X: np.ndarray, y: np.ndarray, measure_name: str
) -> BestSplit:
    if len(np.unique(y)) == 1:
        raise ValueError(
            f"Tried to find a split for homogenous y: {y[:3]} ... {y[-3:]}"
        )

    best_score = None
    best_column = None
    best_threshold = None
    best_target_groups = None

    for array_column in range(X.shape[1]):
        feature_values = X[:, array_column]

        for threshold in feature_values[1:]:
            target_groups = feature_values < threshold
            split_score = SplitScoreMetrics[measure_name](y, target_groups)
            if best_score is None or split_score > best_score:
                best_score = split_score
                best_column = array_column
                best_threshold = threshold
                best_target_groups = target_groups

    return BestSplit(
        float(best_score),
        int(best_column),
        float(best_threshold),
        best_target_groups,
    )


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: str,
    parent_node: Node = None,
    depth: int = 0,
    max_depth: int = None,
    min_improvement: float = 0.0,
) -> Node:
    "Implementation of the Classification And Regression Tree (CART) algorithm"

    n_obs = len(y)

    # check base level because this is a recursive function because tree:
    # * group homogeneous / cannot sensibly be splitted further
    # * no data in the group
    # * max depth reached

    is_baselevel, reason = check_is_baselevel(
        y, parent_node, depth, max_depth=max_depth
    )
    prediction = np.mean(y) if len(y) > 0 else None
    score = (
        SplitScoreMetrics[measure_name](y, np.ones_like(y, dtype=bool))
        if len(y) > 0
        else None
    )
    measure = SplitScore(measure_name, score=score)

    if is_baselevel:
        leaf_node = Node(
            array_column=None,
            threshold=None,
            prediction=prediction,
            left=None,
            right=None,
            measure=measure,
            n_obs=n_obs,
            reason=reason,
        )
        return leaf_node

    # find best split
    best = find_best_split(X, y, measure_name)

    # check if improvement due to split is below minimum requirement
    is_not_good_enough = (
        parent_node is not None
    ) and best.score < parent_node.measure.value + min_improvement
    if is_not_good_enough:
        reason = f"gain due split lower than {min_improvement=}"
        leaf_node = Node(
            array_column=None,
            threshold=None,
            prediction=prediction,
            left=None,
            right=None,
            measure=measure,
            n_obs=n_obs,
            reason=reason,
        )
        return leaf_node

    measure = SplitScore(measure_name, best.score)
    new_node = Node(
        array_column=best.column,
        threshold=best.threshold,
        prediction=prediction,
        left=None,
        right=None,
        measure=measure,
        n_obs=n_obs,
        reason="",
    )

    # descend left
    mask_left = best.target_groups == True
    X_left = X[mask_left, :]
    y_left = y[mask_left]
    new_node.left = grow_tree(
        X_left,
        y_left,
        measure_name=measure_name,
        parent_node=new_node,
        depth=depth + 1,
        max_depth=max_depth,
    )

    # descend right
    mask_right = best.target_groups == False
    X_right = X[mask_right, :]
    y_right = y[mask_right]
    new_node.right = grow_tree(
        X_right,
        y_right,
        measure_name=measure_name,
        parent_node=new_node,
        depth=depth + 1,
        max_depth=max_depth,
    )

    return new_node


def check_if_has_prediction(node: Node) -> bool:
    return node is not None and node.prediction is not None


def find_leaf_node(tree: Node, x: np.ndarray) -> Node:
    node = tree
    while not node.is_leaf:
        go_left = x[node.array_column] < node.threshold
        if go_left:
            if check_if_has_prediction(node.left):
                node = node.left
            else:
                break
        else:
            if check_if_has_prediction(node.right):
                node = node.right
            else:
                break
    return node


def predict_with_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    "Traverse a previously built tree to make one prediction per row in X"

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

    tree_: Node = None

    def __init__(self, max_depth: int, measure_name: str) -> None:
        self.max_depth = max_depth
        self.measure_name = measure_name

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeTemplate":
        raise NotImplementedError()

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class DecisionTreeRegressor(DecisionTreeTemplate):
    """DecisionTreeRegressor

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    tree_: Node = None

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeRegressor":
        X, y = check_X_y(X, y)
        self.tree_ = grow_tree(
            X, y, measure_name=self.measure_name, max_depth=self.max_depth
        )

        return self

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = check_array(X)
        check_is_fitted(self, "tree_")
        y = predict_with_tree(self.tree_, X)

        return y


class DecisionTreeClassifier(DecisionTreeTemplate):
    """DecisionTreeClassifier

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    tree_: Node = None
    classes_: np.ndarray = None

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeClassifier":
        X, y = check_X_y(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree_ = grow_tree(
            X, y, measure_name=self.measure_name, max_depth=self.max_depth
        )

        return self

    def predict_proba(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = check_array(X)
        check_is_fitted(self, "tree_")
        proba = predict_with_tree(self.tree_, X)
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
