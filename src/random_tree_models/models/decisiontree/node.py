import uuid

from pydantic import StrictInt, StrictStr
from pydantic.dataclasses import dataclass

from random_tree_models.models.decisiontree.split_objects import SplitScore


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
