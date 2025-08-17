import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.decisiontree as dtree
import random_tree_models.xgboost as xgboost
from tests.conftest import expected_failed_checks


class TestXGBoostTemplate:
    model = xgboost.XGBoostTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        with pytest.raises(NotImplementedError):
            self.model.fit(None, None)  # type: ignore

    def test_predict(self):
        with pytest.raises(NotImplementedError):
            self.model.predict(None)  # type: ignore


class TestXGBoostRegressor:
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
        model = xgboost.XGBoostRegressor()
        model.fit(self.X, self.y)
        assert all(
            [isinstance(model, dtree.DecisionTreeRegressor) for model in model.trees_]
        )

    def test_predict(self):
        model = xgboost.XGBoostRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestXGBoostClassifier:
    model = xgboost.XGBoostClassifier()

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
        model = xgboost.XGBoostClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert all(
            [isinstance(model, dtree.DecisionTreeRegressor) for model in model.trees_]
        )

    def test_predict(self):
        model = xgboost.XGBoostClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [xgboost.XGBoostRegressor(), xgboost.XGBoostClassifier()],
    expected_failed_checks=expected_failed_checks,  # type: ignore
)
def test_xgboost_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)


@pytest.mark.parametrize(
    "y_float, start_estimate_exp",
    [
        (np.array([-1.0, 1.0]), 0),
        (np.array([-1.0, 1.0, 1.0, 1.0]), 0.5493061443340549),
        (np.array([-1.0, -1.0, -1.0, 1.0]), -0.5493061443340549),
        (np.array([True, True, False, False]), None),
        (np.array([-2.0, -2.0, 2.0, 2.0]), None),
    ],
)
def test_compute_start_estimate_binomial_loglikelihood(
    y_float: np.ndarray, start_estimate_exp: float
):
    try:
        # line to test
        start_estimate = xgboost.compute_start_estimate_binomial_loglikelihood(y_float)
    except ValueError as ex:
        if start_estimate_exp is None:
            pass  # expectedly failed for non -1 and 1 values
        else:
            raise ex
    else:
        if start_estimate_exp is None:
            pytest.fail(f"unexpectedly passed for non -1 and 1 values")
        assert np.isclose(start_estimate, start_estimate_exp)


@pytest.mark.parametrize(
    "y,start_estimate,g_exp",
    [
        (np.array([1.0]), 0.5, np.array([0.5])),
        (np.array([1.0, 1.0]), 0.5, np.array([0.5, 0.5])),
    ],
)
def test_compute_derivatives_negative_least_squares(
    y: np.ndarray, start_estimate: float, g_exp: np.ndarray
):
    # line to test
    g, h = xgboost.compute_derivatives_negative_least_squares(y, start_estimate)

    assert g.shape == h.shape
    assert np.allclose(g, g_exp)
    assert np.allclose(h, -1)


@pytest.mark.parametrize(
    "y_float,start_estimate,g_exp,h_exp",
    [
        (
            np.array([-1.0, 1.0]),
            0.0,
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
        ),
        (
            np.array([-1.0, -1.0, 1.0, 1.0]),
            0.0,
            np.array([-1.0, -1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
        ),
        # failure cases
        (np.array([False, True]), 0.0, None, None),
        (np.array([-2.0, 2.0]), 0.0, None, None),
    ],
)
def test_compute_derivatives_binomial_loglikelihood(
    y_float: np.ndarray,
    start_estimate: float,
    g_exp: np.ndarray,
    h_exp: np.ndarray,
):
    yhat = np.ones_like(y_float) * start_estimate
    is_bad = g_exp is None and h_exp is None
    try:
        # line to test
        g, h = xgboost.compute_derivatives_binomial_loglikelihood(y_float, yhat)
    except ValueError as ex:
        if is_bad:
            pass  # Expectedly failed for incorrect y_float values"
        else:
            raise ex
    else:
        if is_bad:
            pytest.fail("Unexpectedly passed for incorrect y_float values")
        assert g.shape == h.shape
        assert np.allclose(g, g_exp)
        assert np.allclose(h, h_exp)
