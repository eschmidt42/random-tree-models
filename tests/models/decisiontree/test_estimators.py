import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_tree_models.params
from random_tree_models.models.decisiontree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from random_tree_models.models.decisiontree.estimators import DecisionTreeTemplate
from random_tree_models.models.decisiontree.node import Node
from random_tree_models.params import MetricNames
from tests.conftest import expected_failed_checks


class TestDecisionTreeTemplate:
    model = DecisionTreeTemplate(measure_name=MetricNames.entropy)
    X = np.random.normal(size=(100, 10))
    y = np.random.normal(size=(100,))

    def test_tree_(self):
        assert not hasattr(self.model, "tree_")

    def test_growth_params_(self):
        assert not hasattr(self.model, "growth_params_")

        self.model._organize_growth_parameters()
        assert isinstance(
            self.model.growth_params_, random_tree_models.params.TreeGrowthParameters
        )

    def test_fit(self):
        with pytest.raises(NotImplementedError):
            self.model.fit(None, None)  # type: ignore

    def test_predict(self):
        with pytest.raises(NotImplementedError):
            self.model.predict(None)  # type: ignore

    def test_select_samples_and_features_no_sampling(self):
        self.model.frac_features = 1.0
        self.model.frac_subsamples = 1.0
        self.model._organize_growth_parameters()

        # line to test
        X, y, ix_features = self.model._select_samples_and_features(self.X, self.y)

        assert np.allclose(X, self.X)
        assert np.allclose(y, self.y)
        assert np.allclose(ix_features, np.arange(0, self.X.shape[1], 1))

    def test_select_samples_and_features_with_column_sampling(self):
        self.model.frac_features = 0.5
        self.model.frac_subsamples = 1.0
        self.model._organize_growth_parameters()

        # line to test
        X, y, ix_features = self.model._select_samples_and_features(self.X, self.y)

        assert np.isclose(
            X.shape[1], self.X.shape[1] * self.model.frac_features, atol=1
        )
        assert np.isclose(y.shape[0], self.y.shape[0])
        assert all([ix in np.arange(0, self.X.shape[1], 1) for ix in ix_features])

    def test_select_samples_and_features_with_row_sampling(self):
        self.model.frac_features = 1.0
        self.model.frac_subsamples = 0.5
        self.model._organize_growth_parameters()

        # line to test
        X, y, ix_features = self.model._select_samples_and_features(self.X, self.y)

        assert np.isclose(X.shape[0], self.X.shape[0] * self.model.frac_subsamples)
        assert np.isclose(y.shape[0], self.y.shape[0] * self.model.frac_subsamples)
        assert np.allclose(ix_features, np.arange(0, self.X.shape[1], 1))

    def test_select_samples_and_features_with_column_and_row_sampling(self):
        self.model.frac_features = 0.5
        self.model.frac_subsamples = 0.5
        self.model._organize_growth_parameters()

        # line to test
        X, y, ix_features = self.model._select_samples_and_features(self.X, self.y)

        assert np.isclose(
            X.shape[1], self.X.shape[1] * self.model.frac_features, atol=1
        )
        assert np.isclose(X.shape[0], self.X.shape[0] * self.model.frac_subsamples)
        assert np.isclose(y.shape[0], self.y.shape[0] * self.model.frac_subsamples)
        assert all([ix in np.arange(0, self.X.shape[1], 1) for ix in ix_features])

    def test_select_samples_and_features_sampling_reproducibility(self):
        self.model.frac_features = 0.5
        self.model.frac_subsamples = 0.5
        self.model._organize_growth_parameters()

        # line to test
        X0, y0, ix_features0 = self.model._select_samples_and_features(self.X, self.y)
        X1, y1, ix_features1 = self.model._select_samples_and_features(self.X, self.y)

        assert np.allclose(X0, X1)
        assert np.allclose(y0, y1)
        assert np.allclose(ix_features0, ix_features1)

    def test_select_features(self):
        ix_features = np.arange(0, self.X.shape[1], 1)
        _X = self.model._select_features(self.X, ix_features)
        assert np.allclose(_X, self.X)

        ix_features = np.array([0, 1, 2])
        _X = self.model._select_features(self.X, ix_features)
        assert _X.shape[1] == 3


class TestDecisionTreeRegressor:
    model = DecisionTreeRegressor()

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
        model = DecisionTreeRegressor()
        model.fit(self.X, self.y)
        assert isinstance(model.tree_, Node)

    def test_predict(self):
        model = DecisionTreeRegressor()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert np.allclose(predictions, self.y)


class TestDecisionTreeClassifier:
    model = DecisionTreeClassifier()

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
        model = DecisionTreeClassifier()
        model.fit(self.X, self.y)
        assert not hasattr(self.model, "classes_")
        assert isinstance(model.tree_, Node)

    def test_predict(self):
        model = DecisionTreeClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@parametrize_with_checks(
    [DecisionTreeRegressor(), DecisionTreeClassifier()],
    expected_failed_checks=expected_failed_checks,  # type: ignore
)
def test_dtree_estimators_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """

    check(estimator)
