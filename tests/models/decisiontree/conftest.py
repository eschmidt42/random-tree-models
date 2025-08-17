from random_tree_models.models.decisiontree.node import Node

# first value in each tuple is the value to test and the second is the flag indicating if this should work
BOOL_OPTIONS_NONE_OKAY = [(False, True), (True, True), ("blub", False)]
INT_OPTIONS_NONE_OKAY = [(0, True), (None, True), ("blub", False)]
INT_OPTIONS_NONE_NOT_OKAY = [(0, True), (None, False), ("blub", False)]
FLOAT_OPTIONS_NONE_OKAY = [
    (-1.0, True),
    (None, True),
    ("blub", False),
]
FLOAT_OPTIONS_NONE_NOT_OKAY = [
    (-1.0, True),
    (None, False),
    ("blub", False),
]
NODE_OPTIONS_NONE_OKAY = [
    (Node(), True),
    (None, True),
    ("blub", False),
]
STR_OPTIONS_NONE_OKAY = [("blub", True), (None, True), (1.0, False)]
STR_OPTIONS_NONE_NOT_OKAY = [
    ("blub", True),
    (None, False),
    (1, False),
    (1.0, False),
]
