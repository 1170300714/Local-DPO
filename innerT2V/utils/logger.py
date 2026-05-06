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


def _reset_root_logger() -> None:
    with _lock:
        if not _default_handler:
            return
        _get_root_logger().removeHandler(_default_handler)
        _default_handler = None


def set_level(level: int) -> None:
    _configure_root_logger()
    _get_root_logger().setLevel(level)


def set_formatter(formatter: logging.Formatter, handler: Optional[logging.Handler] = None) -> None:
    if handler is None:
        handlers = _get_root_logger().handlers
    else:
        handlers = [handler]
    for handler in handlers:
        handler.setFormatter(formatter)


def reset_formatter(handler: Optional[logging.Handler] = None) -> None:
    if handler is None:
        handlers = _get_root_logger().handlers
    else:
        handlers = [handler]
    for handler in handlers:
        handler.setFormatter(None)


def set_default_formatter(handler: Optional[logging.Handler] = None) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(levelname)s - [%(name)s](%(module)s:L%(lineno)d) - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    set_formatter(formatter, handler=handler)


def add_handler(handler: logging.Handler) -> None:
    _configure_root_logger()
    if handler in _get_root_logger().handlers: return
    _get_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    _configure_root_logger()
    if handler not in _get_root_logger().handlers: return
    _get_root_logger().removeHandler(handler)


def get_logger():
    _configure_root_logger()
    return _get_root_logger()
