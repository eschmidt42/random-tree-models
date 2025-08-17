import logging

import numpy as np
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


def test_vectorize_bool_to_float():
    y = np.array([True, False, True, False])
    res = utils.vectorize_bool_to_float(y)
    assert np.all(res == np.array([1.0, -1.0, 1.0, -1.0]))

    y = np.array([True, False, True, True])
    res = utils.vectorize_bool_to_float(y)
    assert np.all(res == np.array([1.0, -1.0, 1.0, 1.0]))

    y = np.array([False, False, True, False])
    res = utils.vectorize_bool_to_float(y)
    assert np.all(res == np.array([-1.0, -1.0, 1.0, -1.0]))
