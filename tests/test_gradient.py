import math

import numpy as np
import pytest

from random_tree_models.gradient import (
    check_y_float,
    get_pseudo_residual_log_odds,
    get_pseudo_residual_mse,
    get_start_estimate_log_odds,
)


def test_check_y_float():
    # Test case 1: Valid input with only -1 and 1
    y_float = np.array([-1, 1, -1, 1])
    check_y_float(y_float)  # Should not raise an error

    # Test case 2: Valid input with only 1
    y_float = np.array([1, 1, 1, 1])
    check_y_float(y_float)  # Should not raise an error

    # Test case 3: Valid input with only -1
    y_float = np.array([-1, -1, -1, -1])
    check_y_float(y_float)  # Should not raise an error

    # Test case 4: Invalid input with 0
    y_float = np.array([-1, 1, 0, 1])
    with pytest.raises(ValueError):
        check_y_float(y_float)

    # Test case 5: Invalid input with values other than -1 and 1
    y_float = np.array([-1, 1, -2, 2])
    with pytest.raises(ValueError):
        check_y_float(y_float)

    # Test case 6: Invalid input with mixed values
    y_float = np.array([-1, 1, 0.5, -0.5])
    with pytest.raises(ValueError):
        check_y_float(y_float)


def test_get_start_estimate_log_odds():
    # Test case 1: Balanced classes (mean close to 0)
    y = np.array([1, -1, 1, -1])
    actual_start_estimate = get_start_estimate_log_odds(y)
    assert np.isclose(actual_start_estimate, 0.0)

    # Test case 2: All positive class
    y = np.array([1, 1, 1, 1])
    actual_start_estimate = get_start_estimate_log_odds(y)
    assert math.isinf(actual_start_estimate)

    # Test case 3: All negative class
    y = np.array([-1, -1, -1, -1])
    actual_start_estimate = get_start_estimate_log_odds(y)
    assert math.isinf(actual_start_estimate)

    # Test case 4: Unbalanced classes
    y = np.array([1, 1, 1, -1])
    actual_start_estimate = get_start_estimate_log_odds(y)
    v = 0.5493061443340549
    assert np.isclose(actual_start_estimate, v)

    # Test case 5: Another set of unbalanced classes
    y = np.array([1, -1, -1, -1])
    actual_start_estimate = get_start_estimate_log_odds(y)
    assert np.isclose(actual_start_estimate, -v)


def test_get_pseudo_residual_log_odds():
    # Test case 1: Basic test with positive and negative values
    y = np.array([1, -1, 1, -1])
    current_estimates = np.array([0.1, 0.2, -0.1, -0.2])
    expected_residuals_1st = np.array(
        [0.90033201, -1.19737532, 1.09966799, -0.80262468]
    )
    expected_residuals_2nd = np.array(
        [-0.99006629, -0.96104298, -0.99006629, -0.96104298]
    )
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_log_odds(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)

    # Test case 2: current_estimates close to zero
    y = np.array([1, -1])
    current_estimates = np.array([0.001, -0.001])
    expected_residuals_1st = np.array([0.999, -0.999])
    expected_residuals_2nd = np.array([-0.999999, -0.999999])
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_log_odds(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)

    # Test case 3: Larger current_estimates
    y = np.array([1, -1])
    current_estimates = np.array([2, -2])
    expected_residuals_1st = np.array([0.03597242, -0.03597242])
    expected_residuals_2nd = np.array([-0.07065082, -0.07065082])
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_log_odds(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)


def test_get_pseudo_residual_mse():
    y = np.array([1.0, 2.0, 3.0])
    current_estimates = np.array([0.5, 1.0, 2.0])
    expected_residuals_1st = np.array([0.5, 1.0, 1.0])
    expected_residuals_2nd = np.array([-1.0, -1.0, -1.0])
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_mse(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)

    # Test with negative values
    y = np.array([-1.0, -2.0, -3.0])
    current_estimates = np.array([-0.5, -1.0, -2.0])
    expected_residuals_1st = np.array([-0.5, -1.0, -1.0])
    expected_residuals_2nd = np.array([-1.0, -1.0, -1.0])
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_mse(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)

    # Test with zero values
    y = np.array([0.0, 0.0, 0.0])
    current_estimates = np.array([0.0, 0.0, 0.0])
    expected_residuals_1st = np.array([0.0, 0.0, 0.0])
    expected_residuals_2nd = np.array([-1.0, -1.0, -1.0])
    actual_residuals_1st, actual_residuals_2nd = get_pseudo_residual_mse(
        y, current_estimates, True
    )
    assert actual_residuals_2nd is not None
    assert np.allclose(actual_residuals_1st, expected_residuals_1st)
    assert np.allclose(actual_residuals_2nd, expected_residuals_2nd)
