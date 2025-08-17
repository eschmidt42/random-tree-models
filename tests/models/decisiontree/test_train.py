import numpy as np
import pytest
from dirty_equals import IsApprox
from inline_snapshot import snapshot

from random_tree_models.models.decisiontree.node import Node
from random_tree_models.models.decisiontree.split_objects import SplitScore
from random_tree_models.models.decisiontree.train import (
    calc_leaf_weight_and_split_score,
    check_is_baselevel,
    grow_tree,
)
from random_tree_models.params import MetricNames, TreeGrowthParameters


@pytest.mark.parametrize(
    "y, depths",
    [
        (y, depths)
        for y in [(np.array([1, 2]), False), (np.array([]), True)]
        for depths in [(1, 2, False), (2, 2, True), (3, 2, True)]
    ],
)
def test_check_is_baselevel(y, depths):
    y, is_baselevel_exp_y = y
    depth, max_depth, is_baselevel_exp_depth = depths
    is_baselevel_exp = is_baselevel_exp_depth or is_baselevel_exp_y

    # line to test
    is_baselevel, msg = check_is_baselevel(y, depth=depth, max_depth=max_depth)

    assert is_baselevel == is_baselevel_exp
    assert isinstance(msg, str)


def test_calc_leaf_weight_and_split_score():
    y = np.array([True, True, False])
    measure_name = MetricNames.gini
    growth_params = TreeGrowthParameters(max_depth=2)
    g = np.array([1, 2, 3])
    h = np.array([4, 5, 6])

    # line to test
    leaf_weight, split_score = calc_leaf_weight_and_split_score(
        y, measure_name, growth_params, g, h
    )

    assert leaf_weight == IsApprox(0.6666666666666666)
    assert split_score == IsApprox(-0.4444444444444445)


class Test_grow_tree:
    X = np.array([[1], [2], [3]])
    y = np.array([True, True, False])
    target_groups = np.array([True, True, False])
    measure_name = MetricNames.gini
    depth_dummy = 0

    def test_baselevel(self):
        # test returned leaf node
        growth_params = TreeGrowthParameters(max_depth=2)
        parent_node = None

        # line to test
        leaf_node = grow_tree(
            self.X,
            self.y,
            self.measure_name,
            growth_params=growth_params,
            parent_node=parent_node,
            depth=self.depth_dummy,
        )

        assert leaf_node == snapshot(
            Node(
                array_column=0,
                threshold=3.0,
                prediction=0.6666666666666666,
                right=Node(
                    prediction=0.0,
                    measure=SplitScore(name="gini", value=0.0),
                    n_obs=1,
                    reason="homogenous group",
                    depth=1,
                ),
                left=Node(
                    prediction=1.0,
                    measure=SplitScore(name="gini", value=0.0),
                    n_obs=2,
                    reason="homogenous group",
                    depth=1,
                ),
                measure=SplitScore(name="gini", value=0.0),
                n_obs=3,
                reason="",
                depth=0,
            )
        )
