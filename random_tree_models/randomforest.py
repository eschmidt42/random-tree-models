import typing as T

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import base
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import random_tree_models.decisiontree as dtree
import random_tree_models.gradientboostedtrees as gbt


class RandomForestTemplate(base.BaseEstimator):
    def __init__(
        self,
        n_trees: int = 3,
        measure_name: str = None,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        force_all_finite: bool = True,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees
        self.force_all_finite = force_all_finite
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.random_state = random_state

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ):
        raise NotImplementedError()

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class RandomForestRegressor(RandomForestTemplate, base.RegressorMixin):
    """Random forest regressor

    Breiman et al. 2001, Random Forests
    https://doi.org/10.1023/A:1010933404324
    """

    def __init__(self, measure_name: str = "variance", **kwargs) -> None:
        super().__init__(measure_name=measure_name, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X, y = check_X_y(X, y, force_all_finite=self.force_all_finite)
        self.n_features_in_ = X.shape[1]

        self.trees_: T.List[dtree.DecisionTreeRegressor] = []
        rng = np.random.RandomState(self.random_state)
        for _ in track(
            range(self.n_trees), total=self.n_trees, description="tree"
        ):
            # train decision tree to predict differences
            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                force_all_finite=self.force_all_finite,
                frac_subsamples=self.frac_subsamples,
                frac_features=self.frac_features,
                random_state=rng.randint(0, 2**32 - 1),
            )
            new_tree.fit(X, y)
            self.trees_.append(new_tree)

        return self

    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        check_is_fitted(self, ("trees_", "n_features_in_"))
        X = check_array(X, force_all_finite=self.force_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        y = np.zeros((X.shape[0], self.n_trees), dtype=float)

        for i, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):
            y[:, i] = tree.predict(X)

        if aggregation == "mean":
            y = np.mean(y, axis=1)
        elif aggregation == "median":
            y = np.median(y, axis=1)

        return y


class RandomForestClassifier(RandomForestTemplate, base.ClassifierMixin):
    """Random forest classifier

    Breiman et al. 2001, Random Forests
    https://doi.org/10.1023/A:1010933404324
    """

    def __init__(self, measure_name: str = "gini", **kwargs) -> None:
        super().__init__(measure_name=measure_name, **kwargs)

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X, y = check_X_y(X, y, force_all_finite=self.force_all_finite)
        check_classification_targets(y)
        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.trees_: T.List[dtree.DecisionTreeRegressor] = []

        rng = np.random.RandomState(self.random_state)
        for _ in track(
            range(self.n_trees), description="tree", total=self.n_trees
        ):
            new_tree = dtree.DecisionTreeClassifier(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                force_all_finite=self.force_all_finite,
                frac_subsamples=self.frac_subsamples,
                frac_features=self.frac_features,
                random_state=rng.randint(0, 2**32 - 1),
            )
            new_tree.fit(X, y)
            self.trees_.append(new_tree)

        return self

    def predict_proba(
        self, X: T.Union[pd.DataFrame, np.ndarray], aggregation: str = "mean"
    ) -> np.ndarray:
        check_is_fitted(self, ("trees_", "classes_", "n_features_in_"))

        X = check_array(X, force_all_finite=self.force_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        proba = np.zeros(
            (X.shape[0], self.n_trees, len(self.classes_)), dtype=float
        )

        for i, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):
            proba[:, i, :] = tree.predict_proba(X)

        if aggregation == "mean":
            proba = np.mean(proba, axis=1)
            proba = proba / np.sum(proba, axis=1)[:, np.newaxis]
        elif aggregation == "median":
            proba = np.median(proba, axis=1)
            proba = proba / np.sum(proba, axis=1)[:, np.newaxis]

        return proba

    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        proba = self.predict_proba(X, aggregation=aggregation)

        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
