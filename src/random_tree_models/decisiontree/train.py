import typing as T

import numpy as np

import random_tree_models.leafweights as leafweights
import random_tree_models.params
import random_tree_models.scoring as scoring
from random_tree_models.decisiontree.node import Node
from random_tree_models.decisiontree.split import (
    check_if_split_sensible,
    find_best_split,
    select_arrays_for_child_node,
)
from random_tree_models.decisiontree.split_objects import SplitScore
from random_tree_models.params import MetricNames


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


def calc_leaf_weight_and_split_score(
    y: np.ndarray,
    measure_name: random_tree_models.params.MetricNames,
    growth_params: random_tree_models.params.TreeGrowthParameters,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> tuple[float | None, float]:
    leaf_weight = leafweights.calc_leaf_weight(y, measure_name, growth_params, g=g, h=h)

    yhat = leaf_weight * np.ones_like(y)
    score = scoring.calc_split_score(
        measure_name,
        y,
        np.ones_like(y, dtype=bool),
        yhat=yhat,
        g=g,
        h=h,
        growth_params=growth_params,
    )

    return leaf_weight, score


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    measure_name: MetricNames,
    growth_params: random_tree_models.params.TreeGrowthParameters,
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
    if parent_node is None:
        scoring.reset_incrementing_score()

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
