import logging

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
