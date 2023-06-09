import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.decisiontree as dtree
import random_tree_models.extratrees as et


class TestExtraTreesTemplate:
    model = et.ExtraTreesTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        try:
            self.model.fit(None, None)
        except NotImplementedError as ex:
            pytest.xfail("ExtraTreesTemplate.fit expectedly refused call")

    def test_predict(self):
        try:
            self.model.predict(None)
        except NotImplementedError as ex:
            pytest.xfail("ExtraTreesTemplate.predict expectedly refused call")


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
        assert all(
            [
                isinstance(model, dtree.DecisionTreeRegressor)
                for model in model.trees_
            ]
        )

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
            [
                isinstance(model, dtree.DecisionTreeClassifier)
                for model in model.trees_
            ]
        )

    def test_predict(self):
        model = et.ExtraTreesClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@pytest.mark.slow
@parametrize_with_checks([et.ExtraTreesRegressor(), et.ExtraTreesClassifier()])
def test_extratrees_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)
