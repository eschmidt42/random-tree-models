import numpy as np

import random_tree_models.models.isolationforest as iforest
from random_tree_models.params import ThresholdSelectionMethod

rng = np.random.RandomState(42)


class TestIsolationTree:
    X_inlier = rng.normal(size=(100, 2), scale=0.1, loc=0)
    X_outlier = rng.normal(size=(100, 2), scale=0.1, loc=10)

    def test_fit(self):
        model = iforest.IsolationTree()
        model.fit(self.X_inlier)
        assert hasattr(model, "tree_")
        assert hasattr(model, "growth_params_")

    def test_predict(self):
        model = iforest.IsolationTree()
        model.fit(self.X_inlier)

        predictions_inlier = model.predict(self.X_inlier)
        predictions_outlier = model.predict(self.X_outlier)

        assert predictions_inlier.shape == (self.X_inlier.shape[0],)
        assert predictions_outlier.shape == (self.X_outlier.shape[0],)
        mean_path_length_inlier = predictions_inlier.mean()
        mean_path_length_outlier = predictions_outlier.mean()
        assert np.greater_equal(predictions_inlier, 1).all()
        assert np.greater_equal(predictions_outlier, 1).all()
        assert mean_path_length_inlier > mean_path_length_outlier


class TestIsolationForest:
    X_inlier = rng.normal(size=(1000, 2), scale=0.1, loc=0)
    X_outlier = rng.normal(size=(1000, 2), scale=0.1, loc=10)

    def test_fit(self):
        model = iforest.IsolationForest(
            threshold_method=ThresholdSelectionMethod.uniform
        )
        model.fit(self.X_inlier)
        assert hasattr(model, "trees_")

    def test_predict(self):
        model = iforest.IsolationForest(
            threshold_method=ThresholdSelectionMethod.uniform
        )
        model.fit(self.X_inlier)

        predictions_inlier = model.predict(self.X_inlier)
        predictions_outlier = model.predict(self.X_outlier)

        assert predictions_inlier.shape == (self.X_inlier.shape[0],)
        assert predictions_outlier.shape == (self.X_outlier.shape[0],)
        mean_path_length_inlier = predictions_inlier.mean()
        mean_path_length_outlier = predictions_outlier.mean()
        assert np.greater_equal(predictions_inlier, 1).all()
        assert np.greater_equal(predictions_outlier, 1).all()
        assert mean_path_length_inlier > mean_path_length_outlier
