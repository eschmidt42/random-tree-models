import math

import numpy as np


def check_y_float(y_float: np.ndarray):
    # expects y_float to consist only of the values -1 and 1
    unexpected_values = np.abs(y_float) != 1
    if np.sum(unexpected_values) > 0:
        raise ValueError(
            f"expected y_float to contain only -1 and 1, got {y_float[unexpected_values]}"
        )


def get_pseudo_residual_mse(
    y: np.ndarray, current_estimates: np.ndarray, second_order: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    mse loss = sum_i (y_i - estimate_i)^2
    pseudo residual_i = d mse loss(y,estimate) / d estimate_i = - (y_i - estimate_i)
    since we want to apply it as the negative gradient for steepest descent we flip the sign
    """
    first_derivative = y - current_estimates

    second_derivative = None
    if second_order:
        second_derivative = -1 * np.ones_like(first_derivative)

    return first_derivative, second_derivative


def get_pseudo_residual_log_odds(
    y: np.ndarray, current_estimates: np.ndarray, second_order: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    first derivative: d loss / d current_estimates, g in the xgboost paper
    second derivative: d^2 loss / d current_estimates^2, h in the xgboost paper

    """
    check_y_float(y)

    a = np.exp(2 * y * current_estimates)
    first_derivative = 2 * y / (1 + a)

    second_derivative = None
    if second_order:
        second_derivative = -(4 * y**2 * a / (1 + a) ** 2)

    return first_derivative, second_derivative


def get_start_estimate_mse(y: np.ndarray) -> float:
    return float(np.mean(y))


def get_start_estimate_log_odds(y: np.ndarray) -> float:
    """
    1/2 log(1+ym)/(1-ym) because ym is in [-1, 1]
    equivalent to log(ym)/(1-ym) if ym were in [0, 1]
    """
    check_y_float(y)

    ym = np.mean(y)

    if ym == 1:
        return math.inf
    elif ym == -1:
        return -math.inf

    start_estimate = 0.5 * math.log((1 + ym) / (1 - ym))

    return start_estimate
