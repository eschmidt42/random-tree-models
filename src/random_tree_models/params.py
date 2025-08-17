from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, StrictFloat, StrictInt


class ColumnSelectionMethod(StrEnum):
    ascending = "ascending"
    largest_delta = "largest_delta"
    random = "random"


class ThresholdSelectionMethod(StrEnum):
    bruteforce = "bruteforce"
    quantile = "quantile"
    random = "random"
    uniform = "uniform"


class ThresholdSelectionParameters(BaseModel):
    method: ThresholdSelectionMethod
    quantile: StrictFloat = 0.1
    random_state: StrictInt = 0
    n_thresholds: StrictInt = 100
    num_quantile_steps: StrictInt = -1

    def model_post_init(self, context: Any):
        # verify method
        # expected = ThresholdSelectionMethod.__members__.keys()
        # is_okay = self.method in expected
        # if not is_okay:
        #     raise ValueError(
        #         f"passed value for method ('{self.method}') not one of {expected}"
        #     )

        # verify quantile
        is_okay = 0.0 < self.quantile < 1.0
        if not is_okay:
            raise ValueError(f"{self.quantile=} not in (0, 1)")
        is_okay = 1 / self.quantile % 1 == 0
        if not is_okay:
            raise ValueError(f"{self.quantile=} not a valid quantile")

        # verify random_state
        is_okay = self.random_state >= 0
        if not is_okay:
            raise ValueError(f"{self.random_state=} not in [0, inf)")

        # verify n_thresholds valid int
        is_okay = self.n_thresholds > 0
        if not is_okay:
            raise ValueError(f"{self.n_thresholds=} not > 0")

        # set dq
        self.num_quantile_steps = int(1 / self.quantile) + 1


class ColumnSelectionParameters(BaseModel):
    method: ColumnSelectionMethod
    n_trials: StrictInt | None = None


class TreeGrowthParameters(BaseModel):
    max_depth: StrictInt
    min_improvement: StrictFloat = 0.0
    # xgboost lambda - multiplied with sum of squares of leaf weights
    # see Chen et al. 2016 equation 2
    lam: StrictFloat = 0.0
    frac_subsamples: StrictFloat = 1.0
    frac_features: StrictFloat = 1.0
    random_state: StrictInt = 0
    threshold_params: ThresholdSelectionParameters = ThresholdSelectionParameters(
        method=ThresholdSelectionMethod.bruteforce,
        quantile=0.1,
        random_state=0,
        n_thresholds=100,
    )
    column_params: ColumnSelectionParameters = ColumnSelectionParameters(
        method=ColumnSelectionMethod.ascending, n_trials=None
    )

    def model_post_init(self, context: Any):
        # verify frac_subsamples
        is_okay = 0.0 < self.frac_subsamples <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_subsamples=} not in (0, 1]")

        # verify frac_features
        is_okay = 0.0 < self.frac_features <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_features=} not in (0, 1]")


class MetricNames(StrEnum):
    variance = auto()
    entropy = auto()
    entropy_rs = auto()
    gini = auto()
    gini_rs = auto()
    # variance for split score because Friedman et al. 2001 in Algorithm 1
    # step 4 minimize the squared error between actual and predicted dloss/dyhat
    friedman_binary_classification = auto()
    xgboost = auto()
    incrementing = auto()
