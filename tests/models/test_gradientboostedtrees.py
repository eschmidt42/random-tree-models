import math

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.models.gradientboostedtrees as gbt
from random_tree_models.models.decisiontree import (
    DecisionTreeRegressor,
)
from tests.conftest import expected_failed_checks


class TestGradientBoostedTreesTemplate:
    model = gbt.GradientBoostedTreesTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        with pytest.raises(NotImplementedError):
            self.model.fit(None, None)  # type: ignore

    def test_predict(self):
        with pytest.raises(NotImplementedError):
            self.model.predict(None)  # type: ignore


class TestGradientBoostedTreesRegressor:
    X = np.array(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])

    def test_fit(self):
        model = gbt.GradientBoostedTreesRegressor()
        model.fit(self.X, self.y)
        assert all([isinstance(model, DecisionTreeRegressor) for model in model.trees_])

    def test_predict(self):
        model = gbt.GradientBoostedTreesRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestGradientBoostedTreesClassifier:
    model = gbt.GradientBoostedTreesClassifier()

    X = np.array(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    y = np.array([False, False, True, True])

    def test_classes_(self):
        assert not hasattr(self.model, "classes_")

    def test_fit(self):
        model = gbt.GradientBoostedTreesClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert all([isinstance(model, DecisionTreeRegressor) for model in model.trees_])

    def test_predict(self):
        model = gbt.GradientBoostedTreesClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [gbt.GradientBoostedTreesRegressor(), gbt.GradientBoostedTreesClassifier()],
    expected_failed_checks=expected_failed_checks,  # type: ignore
)
def test_gbt_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)


def test_get_pseudo_residual_mse():
    y = np.array([1.0, 2.0, 3.0])
    current_estimates = np.array([0.5, 1.0, 2.0])
    expected_residuals = np.array([0.5, 1.0, 1.0])
    actual_residuals = gbt.get_pseudo_residual_mse(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)

    # Test with negative values
    y = np.array([-1.0, -2.0, -3.0])
    current_estimates = np.array([-0.5, -1.0, -2.0])
    expected_residuals = np.array([-0.5, -1.0, -1.0])
    actual_residuals = gbt.get_pseudo_residual_mse(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)

    # Test with zero values
    y = np.array([0.0, 0.0, 0.0])
    current_estimates = np.array([0.0, 0.0, 0.0])
    expected_residuals = np.array([0.0, 0.0, 0.0])
    actual_residuals = gbt.get_pseudo_residual_mse(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)


def test_get_pseudo_residual_log_odds():
    # Test case 1: Basic test with positive and negative values
    y = np.array([1, -1, 1, -1])
    current_estimates = np.array([0.1, 0.2, -0.1, -0.2])
    expected_residuals = 2 * y / (1 + np.exp(2 * y * current_estimates))
    actual_residuals = gbt.get_pseudo_residual_log_odds(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)

    # Test case 2: y close to zero
    y = np.array([0.001, -0.001])
    current_estimates = np.array([0.5, 0.5])
    expected_residuals = 2 * y / (1 + np.exp(2 * y * current_estimates))
    actual_residuals = gbt.get_pseudo_residual_log_odds(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)

    # Test case 3: current_estimates close to zero
    y = np.array([1, -1])
    current_estimates = np.array([0.001, -0.001])
    expected_residuals = 2 * y / (1 + np.exp(2 * y * current_estimates))
    actual_residuals = gbt.get_pseudo_residual_log_odds(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)

    # Test case 4: Larger current_estimates
    y = np.array([1, -1])
    current_estimates = np.array([2, -2])
    expected_residuals = 2 * y / (1 + np.exp(2 * y * current_estimates))
    actual_residuals = gbt.get_pseudo_residual_log_odds(y, current_estimates)
    assert np.allclose(actual_residuals, expected_residuals)


def test_get_start_estimate_log_odds():
    # Test case 1: Balanced classes (mean close to 0)
    y = np.array([1, -1, 1, -1])
    actual_start_estimate = gbt.get_start_estimate_log_odds(y)
    assert np.isclose(actual_start_estimate, 0.0)

    # Test case 2: All positive class
    y = np.array([1, 1, 1, 1])
    actual_start_estimate = gbt.get_start_estimate_log_odds(y)
    assert math.isinf(actual_start_estimate)

    # Test case 3: All negative class
    y = np.array([-1, -1, -1, -1])
    actual_start_estimate = gbt.get_start_estimate_log_odds(y)
    assert math.isinf(actual_start_estimate)

    # Test case 4: Unbalanced classes
    y = np.array([1, 1, 1, -1])
    ym = np.mean(y)
    actual_start_estimate = gbt.get_start_estimate_log_odds(y)
    v = 0.5493061443340549
    assert np.isclose(actual_start_estimate, v)

    # Test case 5: Another set of unbalanced classes
    y = np.array([1, -1, -1, -1])
    ym = np.mean(y)
    actual_start_estimate = gbt.get_start_estimate_log_odds(y)
    assert np.isclose(actual_start_estimate, -v)
