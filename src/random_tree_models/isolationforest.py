import typing as T

import numpy as np
from rich.progress import track
from sklearn import base
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

from random_tree_models.decisiontree import (
    DecisionTreeTemplate,
    find_leaf_node,
    grow_tree,
)
from random_tree_models.decisiontree.node import Node
from random_tree_models.params import (
    ColumnSelectionMethod,
    MetricNames,
    ThresholdSelectionMethod,
)


def predict_with_isolationtree(tree: Node, X: np.ndarray) -> np.ndarray:
    "Traverse a previously built tree to make one prediction per row in X"
    if not isinstance(tree, Node):
        raise ValueError(
            f"Passed `tree` needs to be an instantiation of Node, got {tree=}"
        )
    n_obs = len(X)
    predictions = np.zeros(X.shape[0], dtype=int)

    for i in range(n_obs):
        leaf_node = find_leaf_node(tree, X[i, :])
        predictions[i] = leaf_node.depth

    return predictions


class IsolationTree(base.OutlierMixin, DecisionTreeTemplate):
    """Isolation tree

    Liu et al. 2006, Isolation Forest, algorithm 2
    https://ieeexplore.ieee.org/abstract/document/4781136
    """

    measure_name: MetricNames
    ensure_all_finite: bool
    max_depth: int

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.incrementing,
        ensure_all_finite: bool = True,
        max_depth: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(measure_name=measure_name, max_depth=max_depth, **kwargs)
        self.ensure_all_finite = ensure_all_finite

    def fit(
        self,
        X: np.ndarray,
        y=None,
        **kwargs,
    ) -> "IsolationTree":
        self._organize_growth_parameters()
        X = validate_data(self, X, ensure_all_finite=False)

        dummy_y = np.arange(X.shape[0], dtype=float)

        _X, _y, self.ix_features_ = self._select_samples_and_features(X, dummy_y)

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

        y = predict_with_isolationtree(self.tree_, _X)

        return y


class IsolationForest(base.BaseEstimator, base.OutlierMixin):
    """Isolation forest

    Liu et al. 2006, Isolation Forest, algorithm 1
    https://ieeexplore.ieee.org/abstract/document/4781136
    """

    measure_name: MetricNames
    n_trees: int
    max_depth: int
    ensure_all_finite: bool
    frac_subsamples: float
    frac_features: float
    threshold_method: ThresholdSelectionMethod
    n_thresholds: int
    column_method: ColumnSelectionMethod
    n_columns_to_try: int
    random_state: int

    def __init__(
        self,
        measure_name: MetricNames = MetricNames.incrementing,
        n_trees: int = 100,
        max_depth: int = 10,
        ensure_all_finite: bool = True,
        frac_subsamples: float = 2 / 3,
        frac_features: float = 1.0,
        threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.random,
        n_thresholds: int = 1,
        column_method: ColumnSelectionMethod = ColumnSelectionMethod.random,
        n_columns_to_try: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.measure_name = measure_name
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.ensure_all_finite = ensure_all_finite
        self.frac_subsamples = frac_subsamples
        self.frac_features = frac_features
        self.threshold_method = threshold_method
        self.n_thresholds = n_thresholds
        self.column_method = column_method
        self.n_columns_to_try = n_columns_to_try
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "IsolationForest":
        X = validate_data(self, X, ensure_all_finite=False)

        self.trees_: T.List[IsolationTree] = []
        rng = np.random.RandomState(self.random_state)
        for _ in track(range(self.n_trees), total=self.n_trees, description="tree"):
            # train decision tree to predict differences
            new_tree = IsolationTree(
                measure_name=self.measure_name,
                max_depth=self.max_depth,
                ensure_all_finite=self.ensure_all_finite,
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

        X = validate_data(self, X, reset=False, ensure_all_finite=False)

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
