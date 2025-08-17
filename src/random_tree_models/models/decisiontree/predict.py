import numpy as np

from random_tree_models.models.decisiontree.node import Node


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
        if node.left is not None:
            node = find_leaf_node(node.left, x)
        else:
            raise ValueError(f"Oddly tried to access node.left even though it is None.")
    else:
        if node.right is not None:
            node = find_leaf_node(node.right, x)
        else:
            raise ValueError(
                f"Oddly tried to access node.right even though it is None."
            )

    return node


def predict_with_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    "Traverse a previously built tree to make one prediction per row in X"
    if not isinstance(tree, Node):
        raise ValueError(
            f"Passed `tree` needs to be an instantiation of Node, got {tree=}"
        )
    n_obs = len(X)
    predictions = []

    for i in range(n_obs):
        leaf_node = find_leaf_node(tree, X[i, :])

        predictions.append(leaf_node.prediction)

    predictions = np.array(predictions)
    return predictions
