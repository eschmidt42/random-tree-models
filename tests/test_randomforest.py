import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.decisiontree as dtree
import random_tree_models.randomforest as rf
from tests.conftest import expected_failed_checks


class TestRandomForestTemplate:
    model = rf.RandomForestTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        try:
            self.model.fit(None, None)  # type: ignore
        except NotImplementedError as ex:
            pytest.xfail("RandomForestTemplate.fit expectedly refused call")

    def test_predict(self):
        try:
            self.model.predict(None)  # type: ignore
        except NotImplementedError as ex:
            pytest.xfail("RandomForestTemplate.predict expectedly refused call")


class TestRandomForestRegressor:
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
        model = rf.RandomForestRegressor()
        model.fit(self.X, self.y)
        assert all(
            [isinstance(model, dtree.DecisionTreeRegressor) for model in model.trees_]
        )

    def test_predict(self):
        model = rf.RandomForestRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestXGBoostClassifier:
    model = rf.RandomForestClassifier()

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
        model = rf.RandomForestClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert all(
            [isinstance(model, dtree.DecisionTreeClassifier) for model in model.trees_]
        )

    def test_predict(self):
        model = rf.RandomForestClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [rf.RandomForestRegressor(), rf.RandomForestClassifier()],
    expected_failed_checks=expected_failed_checks,  # type: ignore
)
def test_randomforest_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)
