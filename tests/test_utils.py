import pytest

import random_tree_models.utils as utils


def test_ColumnSelectionMethod():
    expected = ["ascending", "largest_delta", "random"]
    assert list(utils.ColumnSelectionMethod.__members__.keys()) == expected


def test_ThresholdSelectionMethod():
    expected = ["bruteforce", "quantile", "random", "uniform"]
    assert list(utils.ThresholdSelectionMethod.__members__.keys()) == expected


# method, quantile, random_state, n_thresholds
class TestThresholdSelectionParameters:
    def test_expected_okay(self):
        params = utils.ThresholdSelectionParameters(
            method="quantile", quantile=0.1, random_state=0, n_thresholds=100
        )
        assert params.method == utils.ThresholdSelectionMethod.quantile
        assert params.quantile == 0.1
        assert params.random_state == 0
        assert params.n_thresholds == 100
        assert params.num_quantile_steps == 11

    def test_method_fail(self):
        try:
            _ = utils.ThresholdSelectionParameters(
                method="wuppy", quantile=0.1, random_state=0, n_thresholds=100
            )
        except ValueError as ex:
            pytest.xfail(f"init with unknown method should fail: {ex}")
        else:
            pytest.fail(f"init with unknown method should fail")

    @pytest.mark.parametrize(
        "q,fail",
        [(-0.1, True), (0.0, True), (0.5, False), (1.0, True), (1.1, True)],
    )
    def test_quantile(self, q: float, fail: bool):
        try:
            _ = utils.ThresholdSelectionParameters(
                method="quantile", quantile=q, random_state=0, n_thresholds=100
            )
        except ValueError as ex:
            if fail:
                pytest.xfail(f"init with quantile {q} should fail: {ex}")
            else:
                pytest.fail(f"init with quantile {q} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with quantile {q} should fail: {ex}")

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
            _ = utils.ThresholdSelectionParameters(
                method="quantile",
                quantile=0.1,
                random_state=random_state,
                n_thresholds=100,
            )
        except ValueError as ex:
            if fail:
                pytest.xfail(f"init with {random_state=} should fail: {ex}")
            else:
                pytest.fail(f"init with {random_state=} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with {random_state=} should fail: {ex}")

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
            _ = utils.ThresholdSelectionParameters(
                method="quantile",
                quantile=0.1,
                random_state=42,
                n_thresholds=n_thresholds,
            )
        except ValueError as ex:
            if fail:
                pytest.xfail(f"init with {n_thresholds=} should fail: {ex}")
            else:
                pytest.fail(f"init with {n_thresholds=} should fail: {ex}")
        else:
            if fail:
                pytest.fail(f"init with {n_thresholds=} should fail: {ex}")
