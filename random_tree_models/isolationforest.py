# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import base
from sklearn.utils.validation import check_array, check_is_fitted

import random_tree_models.decisiontree as dtree
import random_tree_models.scoring as scoring


# TODO: add tests
def predict_with_isolationtree(tree: dtree.Node, X: np.ndarray) -> np.ndarray:
    "Traverse a previously built tree to make one prediction per row in X"
    if not isinstance(tree, dtree.Node):
        raise ValueError(
            f"Passed `tree` needs to be an instantiation of Node, got {tree=}"
        )
    n_obs = len(X)
    predictions = np.zeros(X.shape[0], dtype=int)

    for i in range(n_obs):
        leaf_node = dtree.find_leaf_node(tree, X[i, :])
        predictions[i] = leaf_node.depth

    return predictions


class IsolationTree(dtree.DecisionTreeTemplate, base.OutlierMixin):
    """Isolation tree

    Liu et al. 2006, Isolation Forest, algorithm 2
    https://ieeexplore.ieee.org/abstract/document/4781136
    """

    def __init__(
        self,
        force_all_finite: bool = True,
        measure_name: str = "incrementing",
        max_depth: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            measure_name=measure_name, max_depth=max_depth, **kwargs
        )
        self.force_all_finite = force_all_finite

    def fit(
        self,
        X: T.Union[pd.DataFrame, np.ndarray],
        y=None,
        **kwargs,
    ) -> "IsolationTree":
        self._organize_growth_parameters()
        X = check_array(X, force_all_finite=self.force_all_finite)
        self.n_features_in_ = X.shape[1]

        dummy_y = np.arange(X.shape[0], dtype=float)

        _X, _y, self.ix_features_ = self._select_samples_and_features(
            X, dummy_y
        )
        self.incrementing_score_ = scoring.IncrementingScore()

        self.tree_ = dtree.grow_tree(
            _X,
            _y,
            growth_params=self.growth_params_,
            random_state=self.random_state,
            incrementing_score=self.incrementing_score_,
            **kwargs,
        )

        return self

    def predict(self, X: T.Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ("tree_", "growth_params_"))

        X = check_array(X, force_all_finite=self.force_all_finite)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]=} != {self.n_features_in_=}")

        _X = self._select_features(X, self.ix_features_)

        y = predict_with_isolationtree(self.tree_, _X)

        return y


class IsolationForest(base.BaseEstimator, base.OutlierMixin):
    """Isolation forest

    Liu et al. 2006, Isolation Forest, algorithm 1
    https://ieeexplore.ieee.org/abstract/document/4781136
    """

    def __init__(
        self,
        n_trees: int = 100,
        measure_name: str = "incrementing",
        max_depth: int = 10,
        force_all_finite: bool = True,
        frac_subsamples: float = 2 / 3,
        frac_features: float = 1.0,
        threshold_method: str = "random",
        n_thresholds: int = 1,
        column_method: str = "random",
        n_columns_to_try: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.force_all_finite = force_all_finite
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.threshold_method = threshold_method
        self.n_thresholds = n_thresholds
        self.column_method = column_method
        self.n_columns_to_try = n_columns_to_try
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "IsolationForest":
        X = check_array(X, force_all_finite=self.force_all_finite)

        self.n_features_in_ = X.shape[1]

        self.trees_: T.List[IsolationTree] = []
        rng = np.random.RandomState(self.random_state)
        for _ in track(
            range(self.n_trees), total=self.n_trees, description="tree"
        ):
            # train decision tree to predict differences
            new_tree = IsolationTree(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                force_all_finite=self.force_all_finite,
                frac_subsamples=self.frac_subsamples,
                frac_features=self.frac_features,
                threshold_method=self.threshold_method,
                n_thresholds=self.n_thresholds,
                column_method=self.column_method,
                n_columns_to_try=self.n_columns_to_try,
                random_state=rng.randint(0, 2**32 - 1),
            )
            new_tree.fit(X)
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
