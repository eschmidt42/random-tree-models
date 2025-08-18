import logging

import random_tree_models.utils as utils


def test_get_logger():
    logger = utils._get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "rich"
    assert logger.level == logging.INFO
