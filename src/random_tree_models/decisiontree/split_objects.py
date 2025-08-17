import numpy as np
from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr
from pydantic.dataclasses import dataclass


@dataclass(validate_on_init=True)
class SplitScore:
    name: StrictStr  # name of the score used
    value: StrictFloat | None = None  # optimization value gini etc


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BestSplit:
    score: StrictFloat
    column: StrictInt
    threshold: StrictFloat
    target_groups: np.ndarray = Field(default_factory=lambda: np.zeros(10))
    default_is_left: StrictBool | None = None
