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
