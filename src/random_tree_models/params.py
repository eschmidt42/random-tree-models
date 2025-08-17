from enum import StrEnum, auto
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, StrictFloat, StrictInt


class ColumnSelectionMethod(StrEnum):
    ascending = "ascending"
    largest_delta = "largest_delta"
    random = "random"


class ThresholdSelectionMethod(StrEnum):
    bruteforce = "bruteforce"
    quantile = "quantile"
    random = "random"
    uniform = "uniform"


def is_quantile(quantile: float) -> float:
    is_okay = 0.0 < quantile < 1.0
    if not is_okay:
        raise ValueError(f"{quantile=} not in (0, 1)")

    is_okay = 1 / quantile % 1 == 0
    if not is_okay:
        raise ValueError(f"{quantile=} not a valid quantile")

    return quantile


def is_fraction(fraction: float) -> float:
    is_okay = 0.0 < fraction <= 1.0
    if not is_okay:
        raise ValueError(f"{fraction=} not in (0, 1]")

    return fraction


def is_greater_zero(value: int) -> int:
    if value <= 0:
        raise ValueError(f"{value=} not > 0")
    return value


def is_greater_equal_zero(value: int | float) -> int | float:
    if value < 0:
        raise ValueError(f"{value=} not >= 0")
    return value


QuantileValidator = AfterValidator(is_quantile)
GreaterEqualZeroValidator = AfterValidator(is_greater_equal_zero)
GreaterZeroValidator = AfterValidator(is_greater_zero)


class ThresholdSelectionParameters(BaseModel):
    method: ThresholdSelectionMethod

    quantile: Annotated[StrictFloat, QuantileValidator] = 0.1

    random_state: Annotated[StrictInt, GreaterEqualZeroValidator] = 0

    n_thresholds: Annotated[StrictInt, GreaterZeroValidator] = 100
    num_quantile_steps: StrictInt = -1

    def model_post_init(self, context: Any):
        # set dq
        self.num_quantile_steps = int(1 / self.quantile) + 1


class ColumnSelectionParameters(BaseModel):
    method: ColumnSelectionMethod
    n_trials: StrictInt | None = None


FractionValidator = AfterValidator(is_fraction)


class TreeGrowthParameters(BaseModel):
    max_depth: Annotated[StrictInt, GreaterZeroValidator]
    min_improvement: Annotated[StrictFloat, GreaterEqualZeroValidator] = 0.0
    # xgboost lambda - multiplied with sum of squares of leaf weights
    # see Chen et al. 2016 equation 2
    lam: StrictFloat = 0.0

    frac_subsamples: Annotated[StrictFloat, FractionValidator] = 1.0
    frac_features: Annotated[StrictFloat, FractionValidator] = 1.0
    random_state: Annotated[StrictInt, GreaterEqualZeroValidator] = 0
    threshold_params: ThresholdSelectionParameters = ThresholdSelectionParameters(
        method=ThresholdSelectionMethod.bruteforce,
        quantile=0.1,
        random_state=0,
        n_thresholds=100,
    )
    column_params: ColumnSelectionParameters = ColumnSelectionParameters(
        method=ColumnSelectionMethod.ascending, n_trials=None
    )


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
