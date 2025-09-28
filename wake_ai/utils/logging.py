import logging
from typing import Optional

_verbosity_level: int = 0
_created_logger_names: set[str] = set()


def get_logger(name: str, override_level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if override_level is not None:
        logger.setLevel(override_level)
    else:
        _created_logger_names.add(name)
        if _verbosity_level == 0:
            level = logging.WARNING
        elif _verbosity_level == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
        logger.setLevel(level)
    return logger



def set_verbosity_level(level: int) -> None:
    global _verbosity_level
    _verbosity_level = level
    if level == 0:
        level = logging.WARNING
    elif level == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    for name in _created_logger_names:
        get_logger(name).setLevel(level)


def get_verbosity_level() -> int:
    return _verbosity_level
