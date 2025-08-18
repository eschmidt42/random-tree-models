import math
import typing as T

import numpy as np
import sklearn.base as base
from rich.progress import track
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
)
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # type: ignore
)

from random_tree_models.models.decisiontree import (
    DecisionTreeRegressor,
)
from random_tree_models.params import MetricNames, is_greater_zero
from random_tree_models.utils import bool_to_float


class GradientBoostedTreesTemplate(base.BaseEstimator):
    measure_name: MetricNames
    n_trees: int
    max_depth: int
    min_improvement: float
    ensure_all_finite: bool

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.variance,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
    ) -> None:
        self.n_trees = is_greater_zero(n_trees)
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees
        self.ensure_all_finite = ensure_all_finite

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class GradientBoostedTreesRegressor(
    base.RegressorMixin,
    GradientBoostedTreesTemplate,
):
    """OG Gradient boosted trees regressor

    Friedman 2001, Greedy Function Approximation: A Gradient Boosting Machine
    https://www.jstor.org/stable/2699986

    Algorithm 2 (LS_Boost)

    y = our continuous target
    M = number of boosts

    start_estimate = mean(y)
    for m = 1 to M do:
        dy = y - prev_estimate
        new_rho, new_estimator = arg min(rho, estimator) mse(dy, rho*estimator(x))
        new_estimate = prev_estimate + new_rho * new_estimator(x)
        prev_estimate = new_estimate
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.variance,
        factor: float = 1.0,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            n_trees=n_trees,
            max_depth=max_depth,
            min_improvement=min_improvement,
            ensure_all_finite=ensure_all_finite,
        )
        self.factor = factor

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTreesRegressor":
        X, y = validate_data(self, X, y, ensure_all_finite=self.ensure_all_finite)

        self.trees_: list[DecisionTreeRegressor] = []

        self.start_estimate_ = np.mean(y)

        # initial differences to predict
        _y = y - self.start_estimate_

        for _ in track(range(self.n_trees), total=self.n_trees, description="tree"):
            # train decision tree to predict differences
            new_tree = DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
            )
            new_tree.fit(X, _y)
            self.trees_.append(new_tree)

            # update differences to predict
            dy = self.factor * new_tree.predict(X)
            _y = _y - dy

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("trees_", "n_features_in_", "start_estimate_"))

        X = validate_data(
            self, X, reset=False, ensure_all_finite=self.ensure_all_finite
        )

        # baseline estimate
        y = np.ones(X.shape[0]) * self.start_estimate_

        # improve on baseline
        for tree in track(
            self.trees_, description="tree", total=len(self.trees_)
        ):  # loop boosts
            dy = self.factor * tree.predict(X)
            y += dy

        return y


def get_start_estimate(y: np.ndarray) -> float:
    ym = np.mean(y)
    start_estimate = 0.5 * math.log((1 + ym) / (1 - ym))
    return start_estimate


class GradientBoostedTreesClassifier(
    base.ClassifierMixin,
    GradientBoostedTreesTemplate,
):
    """OG gradient boosted trees classifier

    Friedman 2001, Greedy Function Approximation: A Gradient Boosting Machine
    https://www.jstor.org/stable/2699986

    Algorithm 5 (LK_TreeBoost)

    y = our binary target (assumed in paper as -1 or 1)
    M = number of boosts
    loss = log(1+ exp(-2*y*estimate))
    estimate = 1/2 * log (P(y==1)/P(y==-1)) # log odds

    training
    --------

    start_estimate = 1/2 log (P(y==1) / P(y==-1))
    for m = 1 to M do:
        # d loss / d estimate
        dy = 2*y / (1 + exp(2*y*prev_estimate))
        new_estimator = estimator(x,dy)
        # new_estimator leaf sensitive update
        gamma[m,leaf j] = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy))) # looks weird, see comment below*
        new_estimate = prev_estimate + (sum_j gamma[m,leaf j] if x in leaf j)
        prev_estimate = new_estimate

    computing the probability using new_estimate:
    P(y==1 given trees) = 1 / (1 + exp(-2*new_estimate))

    *gamma[m,leaf j] comment:
    Derivation of new_estimate takes three steps:
    1) loss minimal for estimate:
        -> rho_m = arg min_rho (sum_i log(1+exp(-2y*(prev_estimate + rho*h(x)))) )
    2) translation for tree model: rho*h(x) is for a tree just some leaf value gamma
        -> arg min_gamma instead of arg min_rho
        -> gamma[m,leaf j] = arg min_gamma (sum_i log(1+exp(-2y*(prev_estimate + gamma))) )
    3) estimation of gamma in the previous equation, using Newton-Raphson:
        -> gamma_jm = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy)))
        with d loss / d estimate = dy = 2*y / (1 + exp(2*y*prev_estimate))

    inference
    ---------

    y = start_estimate
    for m = 1 to M do:
        # determine leaf for each x
        leaf j = estimator[m](x)
        # for each x retrieve gamma
        y += gamma[m,leaf j]

    y = 1 / (1 + exp(-2*y)) # converting to probability
    """

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.friedman_binary_classification,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        ensure_all_finite: bool = True,
    ) -> None:
        super().__init__(
            measure_name=measure_name,
            n_trees=n_trees,
            max_depth=max_depth,
            min_improvement=min_improvement,
            ensure_all_finite=ensure_all_finite,
        )

    def _bool_to_float(self, y: np.ndarray) -> np.ndarray:
        f = np.vectorize(bool_to_float)
        return f(y)

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTreesClassifier":
        X, y = validate_data(self, X, y, ensure_all_finite=self.ensure_all_finite)

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
        self.trees_: list[DecisionTreeRegressor] = []

        y = self._bool_to_float(y)
        self.start_estimate_ = get_start_estimate(y)

        yhat = np.ones_like(y) * self.start_estimate_

        for _ in track(range(self.n_trees), description="tree", total=self.n_trees):
            g = (
                2 * y / (1 + np.exp(2 * y * yhat))
            )  # dloss/dyhat, g in the xgboost paper

            new_tree = DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
            )
            new_tree.fit(X, y, g=g)
            self.trees_.append(new_tree)

            # update _y
            g_pred = new_tree.predict(X)
            yhat += g_pred

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self, ("trees_", "classes_", "n_features_in_", "start_estimate_")
        )
        X = validate_data(
            self, X, reset=False, ensure_all_finite=self.ensure_all_finite
        )

        g = np.ones(X.shape[0]) * self.start_estimate_

        for _, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):  # loop boosts
            g += tree.predict(X)

        proba = 1 / (1 + np.exp(-2.0 * g))
        proba = np.array([1 - proba, proba]).T
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)

        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
