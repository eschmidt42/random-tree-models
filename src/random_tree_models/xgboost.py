"""XGBoost

Reference: XGBoost: A Scalable Tree Boosting System, Chen et al. 2016
https://dl.acm.org/doi/10.1145/2939672.2939785
http://arxiv.org/abs/1603.02754 (only pre-print has the appendix)

Key aspects:
* regularization / Chen et al. loss
* histogram / percentiles for splits
* sparsity-aware split finding / "default direction" for missing values
"""

import math
import typing as T

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import base
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
)
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # type: ignore
)

import random_tree_models.decisiontree as dtree
import random_tree_models.gradientboostedtrees as gbt
from random_tree_models.scoring import MetricNames


class XGBoostTemplate(base.BaseEstimator):
    measure_name: MetricNames
    n_trees: int
    max_depth: int
    min_improvement: float
    lam: float
    ensure_all_finite: bool
    use_hist: bool
    n_bins: int

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.xgboost,
        n_trees: int = 3,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
        ensure_all_finite: bool = True,
        use_hist: bool = False,
        n_bins: int = 256,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees
        self.lam = lam
        self.ensure_all_finite = ensure_all_finite
        self.use_hist = use_hist
        self.n_bins = n_bins

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def compute_derivatives_negative_least_squares(
    y: np.ndarray, start_estimate: float
) -> T.Tuple[np.ndarray, np.ndarray]:
    "loss = - mean |y-yhat|^2"
    g = y - start_estimate  # 1st order derivative
    h = -1 * np.ones_like(g)  # 2nd order derivative
    return g, h


# TODO: add tests:
# * X_hist is integer based
# * X_hist has the same shape as X
# * X_hist has the same number of unique values as n_bins per column
# * the function fails if X and h are not of the same length
# TODO: add handling of missing values in X
def xgboost_histogrammify_with_h(
    X: np.ndarray, h: np.ndarray, n_bins: int
) -> T.Tuple[np.ndarray, T.List[np.ndarray]]:
    """Converts X into a histogram representation using XGBoost paper eq 8 and 9 using 2nd order gradient statistics as weights"""
    X_hist = np.zeros_like(X, dtype=int)
    all_x_bin_edges = []
    for i in range(X.shape[1]):
        order = np.argsort(X[:, i])
        h_ordered = h[order]
        x_ordered = X[order, i]

        # compute rank using min-max normalization of cumulative sum
        # this deviates from the paper
        rank = h_ordered.cumsum()
        rank = (rank - rank[0]) / (rank[-1] - rank[0])

        rank_bin_edges = np.histogram_bin_edges(rank, bins=n_bins)
        bin_assignments = pd.cut(
            rank, bins=rank_bin_edges, labels=False, include_lowest=True
        )

        x_bin_edges = np.interp(rank_bin_edges, rank, x_ordered)
        all_x_bin_edges.append(x_bin_edges)

        X_hist[order, i] = bin_assignments

    return X_hist, all_x_bin_edges


# TODO: add test that compares the output with that of xgboost_histogrammify_with_h
def xgboost_histogrammify_with_x_bin_edges(
    X: np.ndarray, all_x_bin_edges: T.List[np.ndarray]
) -> np.ndarray:
    """Converts X into a histogram representation using XGBoost paper eq 8 and 9 using 2nd order gradient statistics as weights"""
    X_hist = np.zeros_like(X, dtype=int)

    for i in range(X.shape[1]):
        bin_assignments = pd.cut(
            X[:, i], bins=all_x_bin_edges[i], labels=False, include_lowest=True
        )

        X_hist[:, i] = bin_assignments

    return X_hist


class XGBoostRegressor(base.RegressorMixin, XGBoostTemplate):
    """XGBoost regressor

    Chen et al. 2016, XGBoost: A Scalable Tree Boosting System
    https://dl.acm.org/doi/10.1145/2939672.2939785
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        X, y = validate_data(self, X, y, ensure_all_finite=False)
        # X, y = check_X_y(X, y, ensure_all_finite=self.ensure_all_finite)
        # self.n_features_in_ = X.shape[1]

        self.trees_: T.List[dtree.DecisionTreeRegressor] = []

        self.start_estimate_: float = float(np.mean(y))

        # initial differences to predict using negative squared error loss
        g, h = compute_derivatives_negative_least_squares(y, self.start_estimate_)
        if self.use_hist:
            X_hist, all_x_bin_edges = xgboost_histogrammify_with_h(
                X, h, n_bins=self.n_bins
            )
            self.all_x_bin_edges_ = all_x_bin_edges
            X = X_hist

        for _ in track(range(self.n_trees), total=self.n_trees, description="tree"):
            # train decision tree to predict differences
            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                lam=self.lam,
                ensure_all_finite=self.ensure_all_finite,
            )
            new_tree.fit(X, y, g=g, h=h)
            self.trees_.append(new_tree)

            # update differences to predict
            g -= new_tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("trees_", "n_features_in_", "start_estimate_"))

        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        # baseline estimate
        y = np.ones(X.shape[0]) * self.start_estimate_

        # map to bins
        if self.use_hist:
            X = xgboost_histogrammify_with_x_bin_edges(X, self.all_x_bin_edges_)

        # improve on baseline
        for tree in track(
            self.trees_, description="tree", total=len(self.trees_)
        ):  # loop boosts
            y += tree.predict(X)

        return y


def check_y_float(y_float: np.ndarray):
    # expects y_float to consist only of the values -1 and 1
    unexpected_values = np.abs(y_float) != 1
    if np.sum(unexpected_values) > 0:
        raise ValueError(
            f"expected y_float to contain only -1 and 1, got {y_float[unexpected_values]}"
        )


def compute_start_estimate_binomial_loglikelihood(y_float: np.ndarray) -> float:
    check_y_float(y_float)

    ym = np.mean(y_float)
    start_estimate = 0.5 * math.log((1 + ym) / (1 - ym))
    return start_estimate


def compute_derivatives_binomial_loglikelihood(
    y_float: np.ndarray, yhat: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray]:
    "loss = - sum log(1+exp(2*y*yhat))"
    check_y_float(y_float)
    # differences to predict using binomial log-likelihood (yes, the negative of the negative :P)
    exp_y_yhat = np.exp(2 * y_float * yhat)
    g = 2 * y_float / (1 + exp_y_yhat)  # dloss/dyhat, g in the xgboost paper
    h = -(
        4 * y_float**2 * exp_y_yhat / (1 + exp_y_yhat) ** 2
    )  # d^2loss/dyhat^2, h in the xgboost paper
    return g, h


class XGBoostClassifier(base.ClassifierMixin, XGBoostTemplate):
    """XGBoost classifier

    Chen et al. 2016, XGBoost: A Scalable Tree Boosting System
    https://dl.acm.org/doi/10.1145/2939672.2939785
    """

    def _bool_to_float(self, y: np.ndarray) -> np.ndarray:
        f = np.vectorize(gbt.bool_to_float)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
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

        # self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.trees_: T.List[dtree.DecisionTreeRegressor] = []
        self.gammas_ = []
        self.all_x_bin_edges_ = []

        # convert y from True/False to 1/-1 for binomial log-likelihood
        y = self._bool_to_float(y)

        # initial estimate
        self.start_estimate_ = compute_start_estimate_binomial_loglikelihood(y)
        yhat = np.ones_like(y) * self.start_estimate_

        for _ in track(range(self.n_trees), description="tree", total=self.n_trees):
            g, h = compute_derivatives_binomial_loglikelihood(y, yhat)

            if self.use_hist:
                _X, all_x_bin_edges = xgboost_histogrammify_with_h(
                    X, h, n_bins=self.n_bins
                )
                self.all_x_bin_edges_.append(all_x_bin_edges)
            else:
                _X = X

            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                lam=self.lam,
                ensure_all_finite=self.ensure_all_finite,
            )
            new_tree.fit(_X, y, g=g, h=h)
            self.trees_.append(new_tree)

            # update _y
            yhat += new_tree.predict(X)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("trees_", "classes_", "gammas_", "n_features_in_"))
        X = validate_data(self, X, reset=False, ensure_all_finite=False)

        g = np.ones(X.shape[0]) * self.start_estimate_

        for boost, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):  # loop boosts
            if self.use_hist:
                _X = xgboost_histogrammify_with_x_bin_edges(
                    X, self.all_x_bin_edges_[boost]
                )
            else:
                _X = X

            g += tree.predict(_X)

        proba = 1 / (1 + np.exp(-2.0 * g))
        proba = np.array([1 - proba, proba]).T
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)

        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
