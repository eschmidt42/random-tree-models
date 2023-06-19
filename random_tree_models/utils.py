import logging
from enum import Enum

from pydantic import StrictFloat, StrictInt
from pydantic.dataclasses import dataclass
from rich.logging import RichHandler


# TODO: add tests
class ColumnSelectionMethod(Enum):
    ascending = "ascending"
    largest_delta = "largest_delta"
    random = "random"


# TODO: add tests
class ThresholdSelectionMethod(Enum):
    bruteforce = "bruteforce"
    quantile = "quantile"
    random = "random"
    uniform = "uniform"


# TODO: add tests
@dataclass
class ThresholdSelectionParameters:
    method: ThresholdSelectionMethod
    quantile: StrictFloat = 0.1
    random_state: StrictInt = 0
    n_thresholds: StrictInt = 100

    def __post_init__(self):
        # verify method
        expected = ThresholdSelectionMethod.__members__.keys()
        is_okay = self.method in expected
        if not is_okay:
            raise ValueError(
                f"passed value for method ('{self.method}') not one of {expected}"
            )

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


# TODO: add tests
@dataclass
class ColumnSelectionParameters:
    method: ColumnSelectionMethod
    n_trials: StrictInt = None


# TODO: add tests
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
    threshold_params: ThresholdSelectionParameters = (
        ThresholdSelectionParameters("bruteforce", 0.1, 0, 100)
    )
    column_params: ColumnSelectionParameters = ColumnSelectionParameters(
        ColumnSelectionMethod.ascending, None
    )

    def __post_init__(self):
        # verify frac_subsamples
        is_okay = 0.0 < self.frac_subsamples <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_subsamples=} not in (0, 1]")

        # verify frac_features
        is_okay = 0.0 < self.frac_features <= 1.0
        if not is_okay:
            raise ValueError(f"{self.frac_features=} not in (0, 1]")


# TODO: add tests
def _get_logger(level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(name)s: %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("rich")


logger = _get_logger()
