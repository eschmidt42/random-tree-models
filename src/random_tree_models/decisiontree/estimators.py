import typing as T

import numpy as np
import sklearn.base as base
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

from random_tree_models.decisiontree.node import Node
from random_tree_models.decisiontree.predict import predict_with_tree
from random_tree_models.decisiontree.train import grow_tree
from random_tree_models.params import (
    ColumnSelectionMethod,
    ColumnSelectionParameters,
    MetricNames,
    ThresholdSelectionMethod,
    ThresholdSelectionParameters,
    TreeGrowthParameters,
)


class DecisionTreeTemplate(base.BaseEstimator):
    """Template for DecisionTree classes

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    max_depth: int
    measure_name: MetricNames
    min_improvement: float
    lam: float
    frac_subsamples: float
    frac_features: float
    random_state: int
    threshold_method: ThresholdSelectionMethod
    threshold_quantile: float
    n_thresholds: int
    column_method: ColumnSelectionMethod
    n_columns_to_try: int | None
    ensure_all_finite: bool
    tree_: Node

    def __init__(
        self,
        measure_name: MetricNames,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.bruteforce,
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: ColumnSelectionMethod = ColumnSelectionMethod.ascending,
        n_columns_to_try: int | None = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
    ) -> None:
        # scikit-learn requires we store parameters like this below, instead of directly assigning TreeGrowthParameters
        self.max_depth = max_depth
        self.measure_name = measure_name
        self.min_improvement = min_improvement
        self.lam = lam
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.random_state = random_state
        self.threshold_method = threshold_method
        self.threshold_quantile = threshold_quantile
        self.n_thresholds = n_thresholds
        self.column_method = column_method
        self.n_columns_to_try = n_columns_to_try
        self.ensure_all_finite = ensure_all_finite

    def _organize_growth_parameters(self):
        lam = -abs(self.lam)  # doing this for probably a good reason

        threshold_params = ThresholdSelectionParameters(
            method=self.threshold_method,
            quantile=self.threshold_quantile,
            n_thresholds=self.n_thresholds,
            random_state=self.random_state,
        )

        column_params = ColumnSelectionParameters(
            method=self.column_method,
            n_trials=self.n_columns_to_try,
        )

        self.growth_params_ = TreeGrowthParameters(
            max_depth=self.max_depth,
            min_improvement=self.min_improvement,
            lam=lam,
            frac_subsamples=self.frac_subsamples,
            frac_features=self.frac_features,
            random_state=self.random_state,
            threshold_params=threshold_params,
            column_params=column_params,
        )

    def _select_samples_and_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Sub-samples rows and columns from X and y"

        if not hasattr(self, "growth_params_"):
            raise ValueError(f"Try calling `fit` first.")

        ix = np.arange(len(X))
        rng = np.random.RandomState(self.growth_params_.random_state)

        if self.growth_params_.frac_subsamples < 1.0:
            n_samples = int(self.growth_params_.frac_subsamples * len(X))
            ix_samples = rng.choice(ix, size=n_samples, replace=False)
        else:
            ix_samples = ix

        if self.frac_features < 1.0:
            n_columns = int(X.shape[1] * self.frac_features)
            ix_features = rng.choice(
                np.arange(X.shape[1]),
                size=n_columns,
                replace=False,
            )
        else:
            ix_features = np.arange(X.shape[1])

        _X = X[ix_samples, :]
        _X = _X[:, ix_features]

        _y = y[ix_samples]
        return _X, _y, ix_features

    def _select_features(self, X: np.ndarray, ix_features: np.ndarray) -> np.ndarray:
        return X[:, ix_features]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DecisionTreeTemplate":
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class DecisionTreeRegressor(base.RegressorMixin, DecisionTreeTemplate):
    """DecisionTreeRegressor

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.variance,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.bruteforce,
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: ColumnSelectionMethod = ColumnSelectionMethod.ascending,
        n_columns_to_try: int | None = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            max_depth=max_depth,
            min_improvement=min_improvement,
            lam=lam,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
            n_thresholds=n_thresholds,
            column_method=column_method,
            n_columns_to_try=n_columns_to_try,
            random_state=random_state,
            ensure_all_finite=ensure_all_finite,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> "DecisionTreeRegressor":
        self._organize_growth_parameters()

        X, y = validate_data(self, X, y, ensure_all_finite=False)

        _X, _y, self.ix_features_ = self._select_samples_and_features(X, y)

        self.tree_ = grow_tree(
            _X,
            _y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
            random_state=self.random_state,
            **kwargs,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("tree_", "growth_params_"))

        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        _X = self._select_features(X, self.ix_features_)

        y = predict_with_tree(self.tree_, _X)

        return y


class DecisionTreeClassifier(base.ClassifierMixin, DecisionTreeTemplate):
    """DecisionTreeClassifier

    Based on: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.gini,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        frac_subsamples: float = 1.0,
        frac_features: float = 1.0,
        threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.bruteforce,
        threshold_quantile: float = 0.1,
        n_thresholds: int = 100,
        column_method: ColumnSelectionMethod = ColumnSelectionMethod.ascending,
        n_columns_to_try: int | None = None,
        random_state: int = 42,
        ensure_all_finite: bool = True,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            max_depth=max_depth,
            min_improvement=min_improvement,
            lam=lam,
            frac_subsamples=frac_subsamples,
            frac_features=frac_features,
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
            n_thresholds=n_thresholds,
            column_method=column_method,
            n_columns_to_try=n_columns_to_try,
            random_state=random_state,
        )
        self.ensure_all_finite = ensure_all_finite

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DecisionTreeClassifier":
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

        self._organize_growth_parameters()

        self.classes_, y = np.unique(y, return_inverse=True)

        _X, _y, self.ix_features_ = self._select_samples_and_features(X, y)

        self.tree_ = grow_tree(
            _X,
            _y,
            measure_name=self.measure_name,
            growth_params=self.growth_params_,
            random_state=self.random_state,
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("tree_", "classes_", "growth_params_"))
        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        _X = self._select_features(X, self.ix_features_)

        proba = predict_with_tree(self.tree_, _X)
        proba = np.array([1 - proba, proba]).T
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
