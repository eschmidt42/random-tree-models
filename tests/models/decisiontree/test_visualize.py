import pytest
from inline_snapshot import snapshot
from pytest import CaptureFixture

from random_tree_models.models.decisiontree.estimators import DecisionTreeTemplate
from random_tree_models.models.decisiontree.node import Node
from random_tree_models.models.decisiontree.visualize import show_tree
from random_tree_models.params import MetricNames


@pytest.fixture
def sample_decision_tree() -> DecisionTreeTemplate:
    """Creates a sample decision tree for testing"""
    root = Node(array_column=0, threshold=0.5, n_obs=100)
    root.left = Node(prediction=1.0, n_obs=50, reason="left leaf")
    root.right = Node(array_column=1, threshold=0.2, n_obs=50)
    root.right.left = Node(prediction=2.0, n_obs=20, reason="right-left leaf")
    root.right.right = Node(prediction=3.0, n_obs=30, reason="right-right leaf")

    tree = DecisionTreeTemplate(measure_name=MetricNames.variance, random_state=42)
    tree.tree_ = root
    return tree


def test_show_tree(capsys: CaptureFixture, sample_decision_tree: DecisionTreeTemplate):
    """Tests that show_tree prints the correct tree representation"""
    show_tree(sample_decision_tree)
    captured = capsys.readouterr()

    # a bit brittle, but good enough for now
    assert captured.out == snapshot("""\
Represenation of 🌲 (DecisionTreeTemplate(measure_name=<MetricNames.variance: \n\
'variance'>))
└──  col idx: 0, threshold: 0.500
    ├── (< 0.500) 🍁 # obs: 50, value: 1.000, leaf reason 'left leaf'
    └── (>= 0.500) col idx: 1, threshold: 0.200
        ├── (< 0.200) 🍁 # obs: 20, value: 2.000, leaf reason 'right-left leaf'
        └── (>= 0.200) 🍁 # obs: 30, value: 3.000, leaf reason 'right-right \n\
            leaf'
""")
