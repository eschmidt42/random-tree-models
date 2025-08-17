import logging

import pytest
from pydantic import ValidationError

import random_tree_models.utils as utils
from random_tree_models.params import (
    ColumnSelectionMethod,
    ColumnSelectionParameters,
    ThresholdSelectionMethod,
    ThresholdSelectionParameters,
    TreeGrowthParameters,
)


def test_ColumnSelectionMethod():
    expected = ["ascending", "largest_delta", "random"]
    assert list(ColumnSelectionMethod.__members__.keys()) == expected


def test_ThresholdSelectionMethod():
    expected = ["bruteforce", "quantile", "random", "uniform"]
    assert list(ThresholdSelectionMethod.__members__.keys()) == expected


# method, quantile, random_state, n_thresholds
class TestThresholdSelectionParameters:
    def test_expected_okay(self):
        params = ThresholdSelectionParameters(
            method=ThresholdSelectionMethod.quantile,
            quantile=0.1,
            random_state=0,
            n_thresholds=100,
        )
        assert params.method == ThresholdSelectionMethod.quantile
        assert params.quantile == 0.1
        assert params.random_state == 0
        assert params.n_thresholds == 100
        assert params.num_quantile_steps == 11

    def test_method_fail(self):
        try:
            _ = ThresholdSelectionParameters(
                method="wuppy",  # type: ignore
                quantile=0.1,
                random_state=0,
                n_thresholds=100,
            )
        except ValueError:
            pass  # f"init with unknown method should fail: {ex}"
        else:
            raise

    @pytest.mark.parametrize(
        "q,fail",
        [(-0.1, True), (0.0, True), (0.5, False), (1.0, True), (1.1, True)],
    )
    def test_quantile(self, q: float, fail: bool):
        try:
            _ = ThresholdSelectionParameters(
                method=ThresholdSelectionMethod.quantile,
                quantile=q,
                random_state=0,
                n_thresholds=100,
            )
        except ValueError as ex:
            if fail:
                pass  # f"init with quantile {q} should fail: {ex}"
            else:
                pytest.fail(f"init with quantile {q} should fail: {ex}")
        else:
            if fail:
                raise

    @pytest.mark.parametrize(
        "random_state,fail",
        [
            (-1, True),
            (0, False),
            (1, False),
        ],
    )
    def test_random_state(self, random_state: int, fail: bool):
        try:
            _ = ThresholdSelectionParameters(
                method=ThresholdSelectionMethod.quantile,
                quantile=0.1,
                random_state=random_state,
                n_thresholds=100,
            )
        except ValueError as ex:
            if fail:
                pass  # f"init with {random_state=} should fail: {ex}"
            else:
                pytest.fail(f"init with {random_state=} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with {random_state=} should fail")

    @pytest.mark.parametrize(
        "n_thresholds,fail",
        [
            (-1, True),
            (0, True),
            (
                1,
                False,
            ),
            (10, False),
        ],
    )
    def test_n_thresholds(self, n_thresholds: int, fail: bool):
        try:
            _ = ThresholdSelectionParameters(
                method=ThresholdSelectionMethod.quantile,
                quantile=0.1,
                random_state=42,
                n_thresholds=n_thresholds,
            )
        except ValueError as ex:
            if fail:
                pass  # f"init with {n_thresholds=} should fail: {ex}"
            else:
                pytest.fail(f"init with {n_thresholds=} should fail: {ex}")
        else:
            if fail:
                raise


def test_ColumnSelectionParameters():
    params = ColumnSelectionParameters(method=ColumnSelectionMethod.random, n_trials=10)
    assert params.method == ColumnSelectionMethod.random
    assert params.n_trials == 10


class TestTreeGrowthParameters:
    def test_expected_okay(self):
        params = TreeGrowthParameters(
            max_depth=10,
            min_improvement=0.0,
            lam=0.0,
            frac_subsamples=1.0,
            frac_features=1.0,
            random_state=0,
            threshold_params=ThresholdSelectionParameters(
                method=ThresholdSelectionMethod.quantile,
                quantile=0.1,
                random_state=0,
                n_thresholds=100,
            ),
            column_params=ColumnSelectionParameters(
                method=ColumnSelectionMethod.random, n_trials=10
            ),
        )
        assert params.max_depth == 10
        assert params.min_improvement == 0.0
        assert params.lam == 0.0
        assert params.frac_subsamples == 1.0
        assert params.frac_features == 1.0
        assert params.random_state == 0
        assert isinstance(
            params.threshold_params,
            ThresholdSelectionParameters,
        )
        assert isinstance(params.column_params, ColumnSelectionParameters)

    @pytest.mark.parametrize(
        "frac_subsamples,fail",
        [
            (-0.1, True),
            (0.0, True),
            (0.5, False),
            (1.0, False),
            (1.1, True),
        ],
    )
    def test_frac_subsamples(self, frac_subsamples: float, fail: bool):
        try:
            _ = TreeGrowthParameters(
                max_depth=10,
                frac_subsamples=frac_subsamples,
            )
        except ValueError as ex:
            if fail:
                pass  # f"init with {frac_subsamples=} should fail: {ex}"
            else:
                pytest.fail(f"init with {frac_subsamples=} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with {frac_subsamples=} should fail")

    @pytest.mark.parametrize(
        "frac_features,fail",
        [
            (-0.1, True),
            (0.0, True),
            (0.5, False),
            (1.0, False),
            (1.1, True),
        ],
    )
    def test_frac_features(self, frac_features: float, fail: bool):
        try:
            _ = TreeGrowthParameters(
                max_depth=10,
                frac_features=frac_features,
            )
        except ValueError as ex:
            if fail:
                pass  # f"init with {frac_features=} should fail: {ex}"
            else:
                pytest.fail(f"init with {frac_features=} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with {frac_features=} should fail")

    def test_fail_if_max_depth_missing(self):
        with pytest.raises(ValidationError):
            _ = TreeGrowthParameters()  # type: ignore


def test_get_logger():
    logger = utils._get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "rich"
    assert logger.level == logging.INFO
