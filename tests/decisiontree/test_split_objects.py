import pytest
from pydantic import ValidationError

from random_tree_models.decisiontree.split_objects import SplitScore

from .conftest import FLOAT_OPTIONS_NONE_OKAY, STR_OPTIONS_NONE_NOT_OKAY


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in STR_OPTIONS_NONE_NOT_OKAY
        for value in FLOAT_OPTIONS_NONE_OKAY
    ],
)
def test_SplitScore(name, value):
    name, name_okay = name
    value, value_okay = value
    is_okay = name_okay and value_okay
    is_bad = not is_okay
    try:
        # line to test
        measure = SplitScore(name=name, value=value)
    except ValidationError as ex:
        if is_okay:
            raise ValueError(f"whoops {name=} {value=} failed with {ex}")
        else:
            pass  # SplitScore validation failed as expected
    else:
        if is_bad:
            pytest.fail(
                f"SplitScore test unexpectedly passed for {name=}, {value=}, {name_okay=}, {value_okay=}, {is_okay=}"
            )

        assert hasattr(measure, "name")
        assert hasattr(measure, "value")
