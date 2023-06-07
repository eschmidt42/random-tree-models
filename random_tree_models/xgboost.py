"""XGBoost

Reference: XGBoost: A Scalable Tree Boosting System, Chen et al. 2016

Key aspects to be implemented here:
* regularization / their loss
* histogram / percentiles for splits
* sparsity-aware split finding / "default direction" for missing values
"""
import math
import typing as T

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import base
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import random_tree_models.decisiontree as dtree
import random_tree_models.gradientboostedtrees as gbt

# TODO: implement histogramming

# TODO: implement "default direction"


class XGBoostTemplate(base.BaseEstimator):
    def __init__(
        self,
        n_trees: int = 3,
        measure_name: str = None,
        max_depth: int = 2,
        min_improvement: float = 0.0,
        lam: float = 0.0,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees
        self.lam = lam

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ):
        raise NotImplementedError()

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


def compute_derivatives_negative_least_squares(
    y: np.ndarray, start_estimate: float
) -> T.Tuple[np.ndarray, np.ndarray]:
    "loss = - mean |y-yhat|^2"
    g = y - start_estimate  # 1st order derivative
    h = -1 * np.ones_like(g)  # 2nd order derivative
    return g, h


class XGBoostRegressor(XGBoostTemplate, base.RegressorMixin):
    """XGBoost regressor

    Chen et al. 2016, XGBoost: A Scalable Tree Boosting System
    https://dl.acm.org/doi/10.1145/2939672.2939785
    """

    def __init__(self, measure_name: str = "xgboost", **kwargs) -> None:
        super().__init__(measure_name=measure_name, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.trees_: T.List[dtree.DecisionTreeRegressor] = []

        self.start_estimate_ = np.mean(y)

        # initial differences to predict using negative squared error loss
        g, h = compute_derivatives_negative_least_squares(
            y, self.start_estimate_
        )

        for _ in track(
            range(self.n_trees), total=self.n_trees, description="tree"
        ):
            # train decision tree to predict differences
            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                lam=self.lam,
            )
            new_tree.fit(X, y, g=g, h=h)
            self.trees_.append(new_tree)

            # update differences to predict
            g -= new_tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("trees_", "n_features_in_", "start_estimate_"))
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        # baseline estimate
        y = np.ones(X.shape[0]) * self.start_estimate_

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


class XGBoostClassifier(XGBoostTemplate, base.ClassifierMixin):
    """XGBoost classifier

    Chen et al. 2016, XGBoost: A Scalable Tree Boosting System
    https://dl.acm.org/doi/10.1145/2939672.2939785
    """

    def __init__(self, measure_name: str = "xgboost", **kwargs) -> None:
        super().__init__(measure_name=measure_name, **kwargs)

    def _bool_to_float(self, y: np.ndarray) -> np.ndarray:
        f = np.vectorize(gbt.bool_to_float)
        return f(y)

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.trees_: T.List[dtree.DecisionTreeRegressor] = []
        self.gammas_ = []

        # convert y from True/False to 1/-1 for binomial log-likelihood
        y = self._bool_to_float(y)

        # initial estimate
        self.start_estimate_ = compute_start_estimate_binomial_loglikelihood(y)
        yhat = np.ones_like(y) * self.start_estimate_

        for _ in track(
            range(self.n_trees), description="tree", total=self.n_trees
        ):
            g, h = compute_derivatives_binomial_loglikelihood(y, yhat)

            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
                lam=self.lam,
            )
            new_tree.fit(X, y, g=g, h=h)
            self.trees_.append(new_tree)

            # update _y
            yhat += new_tree.predict(X)

        return self

    def predict_proba(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(
            self, ("trees_", "classes_", "gammas_", "n_features_in_")
        )

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

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
