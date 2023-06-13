from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from pydantic.dataclasses import dataclass


@dataclass
class TreeGrowthParameters:
    max_depth: StrictInt
    min_improvement: StrictFloat = 0.0
    # xgboost lambda - multiplied with sum of squares of leaf weights
    # see Chen et al. 2016 equation 2
    lam: StrictFloat = 0.0
    frac_subsamples: StrictFloat = 1.0
    frac_features: StrictFloat = 1.0
    random_state: StrictInt = 0

    def __post_init__(self):
        is_okay = 0.0 < self.frac_subsamples <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_subsamples=} not in (0, 1]")

        is_okay = 0.0 < self.frac_features <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_features=} not in (0, 1]")
