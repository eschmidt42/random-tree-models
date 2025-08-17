import pytest
from pydantic import ValidationError

from random_tree_models.decisiontree.node import Node
from random_tree_models.decisiontree.split_objects import SplitScore

from .conftest import (
    BOOL_OPTIONS_NONE_OKAY,
    FLOAT_OPTIONS_NONE_OKAY,
    INT_OPTIONS_NONE_OKAY,
    NODE_OPTIONS_NONE_OKAY,
    STR_OPTIONS_NONE_OKAY,
)


@pytest.mark.parametrize(
    "int_val, float_val, node_val, str_val, bool_val",
    [
        (int_val, float_val, node_val, str_val, bool_val)
        for int_val in INT_OPTIONS_NONE_OKAY
        for float_val in FLOAT_OPTIONS_NONE_OKAY
        for node_val in NODE_OPTIONS_NONE_OKAY
        for str_val in STR_OPTIONS_NONE_OKAY
        for bool_val in BOOL_OPTIONS_NONE_OKAY
    ],
)
def test_Node(int_val, float_val, node_val, str_val, bool_val):
    array_column, array_column_okay = int_val
    threshold, threshold_okay = float_val
    prediction, prediction_okay = float_val
    left, left_okay = node_val
    right, right_okay = node_val
    n_obs, n_obs_okay = int_val
    reason, reason_okay = str_val
    default_is_left, default_is_left_okay = bool_val

    is_okay = all(
        [
            array_column_okay,
            threshold_okay,
            prediction_okay,
            left_okay,
            right_okay,
            n_obs_okay,
            reason_okay,
            default_is_left_okay,
        ]
    )
    measure = SplitScore(name="blub", value=1.0)
    try:
        # line to test
        node = Node(
            array_column=array_column,
            threshold=threshold,
            prediction=prediction,
            default_is_left=default_is_left,
            left=left,
            right=right,
            measure=measure,
            n_obs=n_obs,
            reason=reason,
        )
    except ValidationError as ex:
        if is_okay:
            raise ex
        else:
            pytest.xfail("SplitScore validation failed as expected")
    else:
        for att in [
            "array_column",
            "threshold",
            "prediction",
            "default_is_left",
            "left",
            "right",
            "measure",
            "n_obs",
            "reason",
            "node_id",
        ]:
            assert hasattr(node, att), f"{att=} missing in Node"
        assert node.is_leaf == ((left is None) and (right is None)), (
            f"left: {left is None} right: {right is None}"
        )
