import typing as T
import uuid
from enum import Enum
from functools import partial
from typing import Any

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
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import random_tree_models.leafweights as leafweights
import random_tree_models.scoring as scoring
import random_tree_models.utils as utils


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
    default_is_left: bool = None  # default direction is x is nan

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
    y: np.ndarray, node: Node, depth: int, max_depth: int
) -> T.Tuple[bool, str]:
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
    default_is_left: StrictBool = None


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: str,
    yhat: np.ndarray = None,
    g: np.ndarray = None,
    h: np.ndarray = None,
    growth_params: utils.TreeGrowthParameters = None,
) -> BestSplit:
    """ """
    if len(np.unique(y)) == 1:
        raise ValueError(
            f"Tried to find a split for homogenous y: {y[:3]} ... {y[-3:]}"
        )

    best_score = None
    best_column = None
    best_threshold = None
    best_target_groups = None
    best_default_direction_is_left = None

    for array_column in range(X.shape[1]):
        feature_values = X[:, array_column]
        is_missing = np.isnan(feature_values)
        is_finite = np.logical_not(is_missing)

        # if there are no nans
        if is_finite.all():
            for threshold in feature_values[1:]:
                target_groups = feature_values < threshold
                split_score = scoring.SplitScoreMetrics[measure_name](
                    y,
                    target_groups,
                    yhat=yhat,
                    g=g,
                    h=h,
                    growth_params=growth_params,
                )

                if best_score is None or split_score > best_score:
                    best_score = split_score
                    best_column = array_column
                    best_threshold = threshold
                    best_target_groups = target_groups

        else:  # there are nans
            finite_feature_values = feature_values[is_finite]

            for threshold in finite_feature_values[1:]:
                # shuffling all missing left and right and comparing what "default direction" optimizes the loss.
                # so either left of the threshold includes or excludes missing values.
                # based on Chen et al. 2016, XGBoost: A Scalable Tree Boosting System
                # algorithm 3, sparsity-aware split finding
                for default_direction_left in [True, False]:
                    if (
                        default_direction_left
                    ):  # feature value <= threshold or missing
                        target_groups = np.logical_or(
                            feature_values < threshold, is_missing
                        )
                    else:  # feature value <= threshold and finite
                        target_groups = np.logical_and(
                            feature_values < threshold, is_finite
                        )

                    split_score = scoring.SplitScoreMetrics[measure_name](
                        y,
                        target_groups,
                        yhat=yhat,
                        g=g,
                        h=h,
                        growth_params=growth_params,
                    )

                    if best_score is None or split_score > best_score:
                        best_score = split_score
                        best_column = array_column
                        best_threshold = threshold
                        best_target_groups = target_groups
                        best_default_direction_is_left = default_direction_left

    return BestSplit(
        float(best_score),
        int(best_column),
        float(best_threshold),
        best_target_groups,
        best_default_direction_is_left,
    )


def check_if_gain_insufficient(
    best: BestSplit,
    parent_node: Node,
    growth_params: utils.TreeGrowthParameters,
) -> bool:
    "Verifies if split is sensible"
    if parent_node is None or parent_node.measure.value is None:
        return False, None

    gain = best.score - parent_node.measure.value
    is_insufficient_gain = gain < growth_params.min_improvement

    is_all_onesided = (
        best.target_groups.all() or np.logical_not(best.target_groups).all()
    )

    is_insufficient = is_all_onesided or is_insufficient_gain

    return is_insufficient, gain


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: str,
    yhat: np.ndarray = None,
    parent_node: Node = None,
    depth: int = 0,
    growth_params: utils.TreeGrowthParameters = None,
    g: np.ndarray = None,
    h: np.ndarray = None,
    **kwargs,
) -> Node:
    """Implementation of the Classification And Regression Tree (CART) algorithm

    y - what the tree tries to estimate
    yhat - previous best estimate
    g - dloss/dyhat - first derivative of loss(y,yhat)
    h - d^2loss/dyhat^2 - second derivative of loss(y,yhat)

    Some confusion that may arise: This function is used directly for decision trees
    but also for boosting. In boosting what is to be predicted is actually g (dloss/dy_hat)
    of loss(y,yhat). So the leaf weight to use for prediction may need additional transformation.
    But this is handled on the level of the class that called this function, e.g. GradientBoostedTreesClassifier.
    """

    n_obs = len(y)
    is_baselevel, reason = check_is_baselevel(
        y, parent_node, depth, max_depth=growth_params.max_depth
    )

    # leaf_weight, yhat, score = None, None, None
    if n_obs == 0:
        raise ValueError(
            f"Something went wrong. {parent_node=} handed down an empty set of data points."
        )

    leaf_weight = leafweights.calc_leaf_weight(
        y, measure_name, growth_params, g=g, h=h
    )
    yhat = leaf_weight * np.ones_like(y)
    score = scoring.SplitScoreMetrics[measure_name](
        y,
        np.ones_like(y, dtype=bool),
        yhat=yhat,
        g=g,
        h=h,
        growth_params=growth_params,
    )

    measure = SplitScore(measure_name, score=score)

    if is_baselevel:
        leaf_node = Node(
            array_column=None,
            threshold=None,
            prediction=leaf_weight,
            default_is_left=None,
            left=None,
            right=None,
            measure=measure,
            n_obs=n_obs,
            reason=reason,
        )
        return leaf_node

    # find best split
    best = find_best_split(
        X, y, measure_name, g=g, h=h, growth_params=growth_params
    )

    # check if improvement due to split is below minimum requirement
    is_insufficient_gain, gain = check_if_gain_insufficient(
        best, parent_node, growth_params
    )

    if is_insufficient_gain:
        reason = f"gain due split ({gain=}) lower than {growth_params.min_improvement=} or all data points assigned to one side (is left {best.target_groups.mean()=:.2%})"
        leaf_node = Node(
            array_column=None,
            threshold=None,
            prediction=leaf_weight,
            default_is_left=None,
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
        prediction=leaf_weight,
        default_is_left=best.default_is_left,
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
    g_left = g[mask_left] if g is not None else None
    h_left = h[mask_left] if h is not None else None
    new_node.left = grow_tree(
        X_left,
        y_left,
        measure_name=measure_name,
        parent_node=new_node,
        depth=depth + 1,
        growth_params=growth_params,
        g=g_left,
        h=h_left,
    )

    # descend right
    mask_right = best.target_groups == False
    X_right = X[mask_right, :]
    y_right = y[mask_right]
    g_right = g[mask_right] if g is not None else None
    h_right = h[mask_right] if h is not None else None
    new_node.right = grow_tree(
        X_right,
        y_right,
        measure_name=measure_name,
        parent_node=new_node,
        depth=depth + 1,
        growth_params=growth_params,
        g=g_right,
        h=h_right,
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
    ) -> None:
        self.max_depth = max_depth
        self.measure_name = measure_name
        self.min_improvement = min_improvement
        self.lam = lam

    def _organize_growth_parameters(self):
        self.growth_params_ = utils.TreeGrowthParameters(
            max_depth=self.max_depth,
            min_improvement=self.min_improvement,
            lam=self.lam,
        )

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
        force_all_finite: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(measure_name=measure_name, **kwargs)
        self.force_all_finite = force_all_finite

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "DecisionTreeRegressor":
        self._organize_growth_parameters()
        X, y = check_X_y(X, y, force_all_finite=self.force_all_finite)
        self.n_features_in_ = X.shape[1]
        self.tree_ = grow_tree(
            X,
            y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
            **kwargs,
        )

        return self

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ("tree_", "growth_params_"))

        X = check_array(X, force_all_finite=self.force_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        y = predict_with_tree(self.tree_, X)

        return y


class DecisionTreeClassifier(base.ClassifierMixin, DecisionTreeTemplate):
    """DecisionTreeClassifier

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: str = "gini",
        force_all_finite: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(measure_name=measure_name, **kwargs)
        self.force_all_finite = force_all_finite

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ) -> "DecisionTreeClassifier":
        X, y = check_X_y(X, y, force_all_finite=self.force_all_finite)
        check_classification_targets(y)
        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self._organize_growth_parameters()

        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)

        self.tree_ = grow_tree(
            X,
            y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
        )

        return self

    def predict_proba(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ("tree_", "classes_", "growth_params_"))

        X = check_array(X, force_all_finite=self.force_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        proba = predict_with_tree(self.tree_, X)
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
