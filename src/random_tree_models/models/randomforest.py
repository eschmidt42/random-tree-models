import typing as T

import numpy as np
from rich.progress import track
from sklearn import base
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # type: ignore
)

from random_tree_models.models.decisiontree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from random_tree_models.params import MetricNames, is_greater_zero


class RandomForestTemplate(base.BaseEstimator):
    measure_name: MetricNames
    n_trees: int
    max_depth: int
    min_improvement: float
    ensure_all_finite: bool
    frac_subsamples: float
    frac_features: float
    random_state: int

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.variance,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_trees = is_greater_zero(n_trees)
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees
        self.ensure_all_finite = ensure_all_finite
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class RandomForestRegressor(base.RegressorMixin, RandomForestTemplate):
    """Random forest regressor

    Breiman et al. 2001, Random Forests
    https://doi.org/10.1023/A:1010933404324
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.variance,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            n_trees=n_trees,
            max_depth=max_depth,
            min_improvement=min_improvement,
            ensure_all_finite=ensure_all_finite,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X, y = validate_data(self, X, y, ensure_all_finite=False)

        self.trees_: list[DecisionTreeRegressor] = []
        rng = np.random.RandomState(self.random_state)
        for _ in track(range(self.n_trees), total=self.n_trees, description="tree"):
            new_tree = DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                ensure_all_finite=self.ensure_all_finite,
                frac_subsamples=self.frac_subsamples,
                frac_features=self.frac_features,
                random_state=rng.randint(0, 2**32 - 1),
            )
            new_tree.fit(X, y)
            self.trees_.append(new_tree)

        return self

    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        check_is_fitted(self, ("trees_", "n_features_in_"))

        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        y = np.zeros((X.shape[0], self.n_trees), dtype=float)

        for i, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):
            y[:, i] = tree.predict(X)

        match aggregation:
            case "mean":
                y = np.mean(y, axis=1)
            case "median":
                y = np.median(y, axis=1)
            case _:
                raise ValueError(f"{aggregation=} expected to be 'mean' or 'median'")

        return y


class RandomForestClassifier(base.ClassifierMixin, RandomForestTemplate):
    """Random forest classifier

    Breiman et al. 2001, Random Forests
    https://doi.org/10.1023/A:1010933404324
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.gini,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            n_trees=n_trees,
            max_depth=max_depth,
            min_improvement=min_improvement,
            ensure_all_finite=ensure_all_finite,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            random_state=random_state,
        )

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def __sklearn_tags__(self):
        # https://scikit-learn.org/stable/developers/develop.html
        tags = super().__sklearn_tags__()  # type: ignore
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X, y = validate_data(self, X, y, ensure_all_finite=False)
        check_classification_targets(y)

        y_type = type_of_target(y, input_name="y", raise_unknown=True)  # type: ignore
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )

        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self.classes_, y = np.unique(y, return_inverse=True)
        self.trees_: list[DecisionTreeClassifier] = []

        rng = np.random.RandomState(self.random_state)
        for _ in track(range(self.n_trees), description="tree", total=self.n_trees):
            new_tree = DecisionTreeClassifier(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                ensure_all_finite=self.ensure_all_finite,
                frac_subsamples=self.frac_subsamples,
                frac_features=self.frac_features,
                random_state=rng.randint(0, 2**32 - 1),
            )
            new_tree.fit(X, y)
            self.trees_.append(new_tree)

        return self

    def predict_proba(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        check_is_fitted(self, ("trees_", "classes_", "n_features_in_"))

        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        proba = np.zeros((X.shape[0], self.n_trees, len(self.classes_)), dtype=float)

        for i, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):
            proba[:, i, :] = tree.predict_proba(X)

        match aggregation:
            case "mean":
                proba = np.mean(proba, axis=1)
                proba = proba / np.sum(proba, axis=1)[:, np.newaxis]
            case "median":
                proba = np.median(proba, axis=1)
                proba = proba / np.sum(proba, axis=1)[:, np.newaxis]
            case _:
                raise ValueError(f"{aggregation=} expected to be 'mean' or 'median'")

        return proba

    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        proba = self.predict_proba(X, aggregation=aggregation)

        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
