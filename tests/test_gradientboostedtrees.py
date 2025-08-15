import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.decisiontree as dtree
import random_tree_models.gradientboostedtrees as gbt


class TestGradientBoostedTreesTemplate:
    model = gbt.GradientBoostedTreesTemplate()

    def test_tree_(self):
        assert not hasattr(self.model, "trees_")

    def test_fit(self):
        try:
            self.model.fit(None, None)
        except NotImplementedError as ex:
            pytest.xfail("DecisionTreeTemplate.fit expectedly refused call")

    def test_predict(self):
        try:
            self.model.predict(None)
        except NotImplementedError as ex:
            pytest.xfail("DecisionTreeTemplate.predict expectedly refused call")


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
        assert all(
            [isinstance(model, dtree.DecisionTreeRegressor) for model in model.trees_]
        )

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
        assert all(
            [isinstance(model, dtree.DecisionTreeRegressor) for model in model.trees_]
        )

    def test_predict(self):
        model = gbt.GradientBoostedTreesClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [gbt.GradientBoostedTreesRegressor(), gbt.GradientBoostedTreesClassifier()]
)
def test_gbt_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)


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
        res = gbt.bool_to_float(x)
    except ValueError as ex:
        if is_bad:
            pytest.xfail("Failed expectedly to convert non-bool values")
    else:
        if is_bad:
            pytest.fail(f"Passed unexpectedly for non-bool value {x} returning {res}")
        assert res == exp
