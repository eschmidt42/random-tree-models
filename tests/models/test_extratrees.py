import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.models.extratrees as et
from random_tree_models.models.decisiontree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from random_tree_models.params import MetricNames
from tests.conftest import expected_failed_checks


class TestExtraTreesTemplate:
    model = et.ExtraTreesTemplate(measure_name=MetricNames.gini)

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        with pytest.raises(NotImplementedError):
            self.model.fit(None, None)  # type: ignore

    def test_predict(self):
        with pytest.raises(NotImplementedError):
            self.model.predict(None)  # type: ignore


class TestExtraTreesRegressor:
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
        model = et.ExtraTreesRegressor()
        model.fit(self.X, self.y)
        assert all([isinstance(model, DecisionTreeRegressor) for model in model.trees_])

    def test_predict(self):
        model = et.ExtraTreesRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestXGBoostClassifier:
    model = et.ExtraTreesClassifier()

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
        model = et.ExtraTreesClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert all(
            [isinstance(model, DecisionTreeClassifier) for model in model.trees_]
        )

    def test_predict(self):
        model = et.ExtraTreesClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [et.ExtraTreesRegressor(), et.ExtraTreesClassifier()],
    expected_failed_checks=expected_failed_checks,  # type: ignore
)
def test_extratrees_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)
