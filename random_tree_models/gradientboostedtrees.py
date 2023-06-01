import math
import typing as T

import numpy as np
import pandas as pd
import sklearn.base as base
from rich.progress import track
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import random_tree_models.decisiontree as dtree


class GradientBoostedTreesTemplate(base.BaseEstimator):
    def __init__(
        self,
        n_trees: int = 3,
        measure_name: str = None,
        max_depth: int = 2,
        min_improvement: float = 0.0,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.n_trees = n_trees

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y: T.Union[pd.Series, np.ndarray],
    ):
        raise NotImplementedError()

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class GradientBoostedTreesRegressor(
    GradientBoostedTreesTemplate, base.RegressorMixin
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
        self, factor: float = 1.0, measure_name: str = "variance", **kwargs
    ) -> None:
        super().__init__(measure_name=measure_name, **kwargs)
        self.factor = factor

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "GradientBoostedTreesRegressor":
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.trees_: T.List[dtree.DecisionTreeRegressor] = []

        self.start_estimate_ = np.mean(y)

        # initial differences to predict
        _y = y - self.start_estimate_

        for _ in track(
            range(self.n_trees), total=self.n_trees, description="tree"
        ):
            # train decision tree to predict differences
            new_tree = dtree.DecisionTreeRegressor(
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
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        # baseline estimate
        y = np.ones(X.shape[0]) * self.start_estimate_

        # improve on baseline
        for tree in track(
            self.trees_, description="tree", total=len(self.trees_)
        ):  # loop boosts
            dy = self.factor * tree.predict(X)
            y += dy

        return y


def bool_to_float(x: bool) -> float:
    if x == True:
        return 1.0
    elif x == False:
        return -1.0
    else:
        raise ValueError(f"{x=}, expected bool")


class GradientBoostedTreesClassifier(
    GradientBoostedTreesTemplate, base.ClassifierMixin
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

    def __init__(self, measure_name: str = "variance", **kwargs) -> None:
        super().__init__(measure_name=measure_name, **kwargs)

    def _bool_to_float(self, y: np.ndarray) -> np.ndarray:
        f = np.vectorize(bool_to_float)
        return f(y)

    def _more_tags(self) -> T.Dict[str, bool]:
        """Describes to scikit-learn parametrize_with_checks the scope of this class

        Reference: https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        """
        return {"binary_only": True}

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "GradientBoostedTreesClassifier":
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if len(np.unique(y)) == 1:
            raise ValueError("Cannot train with only one class present")

        self.n_features_in_ = X.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.trees_: T.List[dtree.DecisionTreeRegressor] = []
        self.gammas_ = []

        y = self._bool_to_float(y)
        ym = np.mean(y)
        self.start_estimate_ = 0.5 * math.log((1 + ym) / (1 - ym))

        _y = np.ones_like(y) * self.start_estimate_

        for _ in track(
            range(self.n_trees), description="tree", total=self.n_trees
        ):
            dy = 2 * y / (1 + np.exp(2 * y * _y))

            new_tree = dtree.DecisionTreeRegressor(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                min_improvement=self.min_improvement,
            )
            new_tree.fit(X, dy)
            self.trees_.append(new_tree)

            # collect node ids for each x
            leaf_nodes = [dtree.find_leaf_node(new_tree.tree_, x) for x in X]
            ids = np.array([leaf.node_id for leaf in leaf_nodes])

            # log node ids and corresponding update gamma
            gammas = {}
            for _id in ids:
                # dy only for current id
                dy_for_id = np.where(ids == _id, dy, 0)
                # gamma[m,leaf j] = (sum_i dy) / (sum_i abs(dy)*(2-abs(dy)))
                dy_id_filtered_abs = np.abs(dy_for_id)
                gamma = (
                    dy_for_id.sum()
                    / (dy_id_filtered_abs * (2 - dy_id_filtered_abs)).sum()
                )
                # store
                gammas[_id] = gamma
            # store
            self.gammas_.append(gammas)

            # update _y
            dy = np.array([gammas[_id] for _id in ids])
            _y = _y + dy

        return self

    def predict_proba(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(
            self, ("trees_", "classes_", "gammas_", "n_features_in_")
        )

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        y = np.ones(X.shape[0]) * self.start_estimate_

        for m, tree in track(
            enumerate(self.trees_), description="tree", total=len(self.trees_)
        ):  # loop boosts
            leaf_nodes = [dtree.find_leaf_node(tree.tree_, x) for x in X]
            ids = np.array([leaf.node_id for leaf in leaf_nodes])
            dy = np.array([self.gammas_[m][_id] for _id in ids])
            y += dy

        proba = 1 / (1 + np.exp(-2.0 * y))
        proba = np.array([1 - proba, proba]).T
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)

        ix = np.argmax(proba, axis=1)
        y = self.classes_[ix]

        return y
