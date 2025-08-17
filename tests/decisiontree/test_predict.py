import numpy as np
import pytest

from random_tree_models.decisiontree.node import Node
from random_tree_models.decisiontree.predict import find_leaf_node, predict_with_tree


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
    tree = Node(
        array_column=0,
        threshold=0.0,
        left=Node(
            array_column=1,
            threshold=0.0,
            left=Node(prediction=0.0),
            right=Node(prediction=3.0),
        ),
        right=Node(
            array_column=1,
            threshold=0.0,
            left=Node(prediction=1.0),
            right=Node(prediction=2.0),
        ),
    )
    # line to test
    leaf = find_leaf_node(tree, x)

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
    tree = Node(
        array_column=0,
        threshold=0.0,
        left=Node(
            array_column=1,
            threshold=0.0,
            left=Node(prediction=0.0),
            right=Node(prediction=3.0),
        ),
        right=Node(
            array_column=1,
            threshold=0.0,
            left=Node(prediction=1.0),
            right=Node(prediction=2.0),
        ),
    )

    # line to test
    predictions = predict_with_tree(tree, X)

    assert np.allclose(predictions, np.arange(0, 4, 1))
