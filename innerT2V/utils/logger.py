import sys
import logging
import threading
from typing import Optional


_lock = threading.Lock()
_default_name: str = 'innerT2V'
_default_handler: Optional[logging.Handler] = None


def _get_root_logger() -> logging.Logger:
    return logging.getLogger(_default_name)


def _configure_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler(sys.stdout)

        root_logger = _get_root_logger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        root_logger.addHandler(_default_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False


def set_default_formatter(handler: Optional[logging.Handler] = None) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(levelname)s - [%(name)s](%(module)s:L%(lineno)d) - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    if handler is None:
        handlers = _get_root_logger().handlers
    else:
        handlers = [handler]
    for h in handlers:
        h.setFormatter(formatter)


def add_handler(handler: logging.Handler) -> None:
    _configure_root_logger()
    if handler in _get_root_logger().handlers: return
    _get_root_logger().addHandler(handler)


def get_logger():
    _configure_root_logger()
    return _get_root_logger()
