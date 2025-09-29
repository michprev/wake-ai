import logging
from typing import Optional

_verbosity_level: int = 0
_created_logger_names: set[str] = set()
_verbose_filter: set[str] | None = None


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


def set_verbose_filter(filter: set[str]) -> None:
    global _verbose_filter
    _verbose_filter = filter

def get_verbose_filter() -> set[str] | None:
    return _verbose_filter


def get_verbosity_level() -> int:
    return _verbosity_level


def should_verbose_log(log_type: str) -> bool:
    if _verbose_filter is None:
        return True
    return log_type in _verbose_filter
