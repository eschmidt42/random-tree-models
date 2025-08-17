import logging

import numpy as np
from rich.logging import RichHandler


def _get_logger(level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(name)s: %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("rich")
    logger.setLevel(level)
    return logger


logger = _get_logger()


def bool_to_float(x: bool) -> float:
    if x == True:
        return 1.0
    elif x == False:
        return -1.0
    else:
        raise ValueError(f"{x=}, expected bool")


def vectorize_bool_to_float(y: np.ndarray) -> np.ndarray:
    f = np.vectorize(bool_to_float)
    return f(y)
