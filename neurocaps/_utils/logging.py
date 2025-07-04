"""
Internal logging function and class for flushing. Logger configured to respect user logging
configuration prior to package import while defaulting to console logging if no configurations are
provided.
"""

import logging, sys
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue
from typing import Union

# Global variables to determine if a handler is user defined or defined by OS
_PARALLEL_MODULE = "neurocaps.extraction._internals.postprocess"
_USER_ROOT_HANDLER = None
_USER_MODULE_HANDLERS = {}


class Flush(logging.StreamHandler):
    """Flush logs immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logger(
    name: str,
    flush: bool = False,
    top_level: bool = True,
    parallel_log_config: Union[dict[str, Union[Queue, int]], None] = None,
):
    """
    Generates module specific loggers, defaults to outputting logs at the informational level to
    the standard output. The ``top_level`` parameter is used to determine if logging configurations
    are present at the time of package import to prevent logging's from sending logs to standard
    error by default.

    Notes
    -----
    **Parallel Processing:** When parallel processing using joblib's loky backend with certain
    modules such as "neurocaps._utils.extraction.extract_timeseries", the logger needs to be
    passed to each child process to ensure to ensure logging configurations are respected. Hence in
    the "extract_timeseries", this function is also called inside ``_extract_timeseries`` function.

    To ensure that there is no issue with race conditions when intending to log to files,
    ``parallel_log_config`` is offered to allow for controlled queue-based logging with
    ``QueueHandler``. This requires a centralized queue with a manager object that is able to share
    reference objects of the centralized queue with each child process. Then ``QueueListener`` can
    be set up in the main process to listen to the centralized queue object and write messages to
    the specified file.
    """
    global _USER_ROOT_HANDLER, _USER_MODULE_HANDLERS

    logger = logging.getLogger(name)

    if top_level is True:
        _USER_ROOT_HANDLER = logging.getLogger().hasHandlers()
        _USER_MODULE_HANDLERS[logger.name] = logger.handlers

    # Special case for parallel config to pass a user-defined logger specifically for parallel
    # processing.
    if logger.name == _PARALLEL_MODULE and not top_level and parallel_log_config:
        logger = setup_queuehandler(logger, parallel_log_config)

    if not logger.level:
        logger.setLevel(logging.INFO)

    # Check if user defined root handler or assigned a specific handler for module
    default_handlers = _USER_ROOT_HANDLER or _USER_MODULE_HANDLERS[logger.name]

    # Propagate to root if no user-defined module or a root handler is detected
    logger.propagate = (
        False if _USER_MODULE_HANDLERS[logger.name] or not _USER_ROOT_HANDLER else True
    )

    # Add or messages will repeat several times due to multiple handlers if same name used
    if not default_handlers and not (
        parallel_log_config or (logger.name == _PARALLEL_MODULE and top_level)
    ):
        logger = add_handler(logger, flush)

    return logger


def setup_queuehandler(logger: logging.Logger, parallel_log_config: dict[str, Union[Queue, int]]):
    # Only QueueHandler will be in handler list
    logger.handlers.clear()
    queue = parallel_log_config.get("queue")
    # Non-strict check
    if not queue:
        ValueError("'queue' is a mandatory key and must contain a queue with a manager object.")
    else:
        logger.addHandler(QueueHandler(queue))

    if "level" in parallel_log_config:
        logger.setLevel(parallel_log_config["level"])

    return logger


def add_handler(logger: logging.Logger, flush: bool, format: Union[str, None] = None):
    """Add and format handler."""
    # Safeguard; ensure a clean state for "extract_timeseries" since it is used in parallel and
    # sequential contexts
    if logger.name == _PARALLEL_MODULE:
        logger.handlers.clear()

    if flush:
        handler = Flush(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)

    format = format if format else "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)

    return logger
