import logging

import pytest

import random_tree_models.utils
import random_tree_models.utils as utils


def test_get_logger():
    logger = utils._get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "rich"
    assert logger.level == logging.INFO


@pytest.mark.parametrize(
    "x,exp,is_bad",
    [
        (True, 1, False),
        (False, -1, False),
        ("a", None, True),
        (1, 1, False),
        (0, -1, False),
        (-1, None, True),
        (None, None, True),
    ],
)
def test_bool_to_float(x, exp, is_bad: bool):
    try:
        # line to test
        res = random_tree_models.utils.bool_to_float(x)
    except ValueError as ex:
        if is_bad:
            pass  # Failed expectedly to convert non-bool values
    else:
        if is_bad:
            pytest.fail(f"Passed unexpectedly for non-bool value {x} returning {res}")
        assert res == exp
