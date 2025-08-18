import numpy as np
import pytest

from random_tree_models.transform import (
    bool_to_float,
    get_probabilities_from_mapped_bools,
    vectorize_bool_to_float,
)


def test_vectorize_bool_to_float():
    y = np.array([True, False, True, False])
    res = vectorize_bool_to_float(y)
    assert np.all(res == np.array([1.0, -1.0, 1.0, -1.0]))

    y = np.array([True, False, True, True])
    res = vectorize_bool_to_float(y)
    assert np.all(res == np.array([1.0, -1.0, 1.0, 1.0]))

    y = np.array([False, False, True, False])
    res = vectorize_bool_to_float(y)
    assert np.all(res == np.array([-1.0, -1.0, 1.0, -1.0]))


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
        res = bool_to_float(x)
    except ValueError as ex:
        if is_bad:
            pass  # Failed expectedly to convert non-bool values
    else:
        if is_bad:
            pytest.fail(f"Passed unexpectedly for non-bool value {x} returning {res}")
        assert res == exp


def test_get_probabilities_from_mapped_bools():
    h = np.array([0.0, 1.0, -1.0])
    actual = get_probabilities_from_mapped_bools(h)
    expected = np.array(
        [[0.5, 0.5], [0.11920292, 0.88079708], [0.88079708, 0.11920292]]
    )
    assert np.allclose(actual, expected)

    h = np.array([0.5, -0.5, 0.2])
    actual = get_probabilities_from_mapped_bools(h)
    expected = np.array(
        [[0.26894142, 0.73105858], [0.73105858, 0.26894142], [0.40131234, 0.59868766]]
    )
    assert np.allclose(actual, expected)
