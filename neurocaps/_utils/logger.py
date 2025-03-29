"""
Internal logging function and class for flushing. Logger configured to respect user logging configuration
prior to package import while defaulting to console logging if no configurations are provided.
"""

import logging, sys
from logging.handlers import QueueHandler

# Global variables to determine if a handler is user defined or defined by OS
_USER_ROOT_HANDLER = None
_USER_MODULE_HANDLERS = {}


class _Flush(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _logger(name, flush=False, top_level=True, parallel_log_config=None):
    global _USER_ROOT_HANDLER, _USER_MODULE_HANDLERS

    logger = logging.getLogger(name)
    parallel_module = "neurocaps._utils.extraction.extract_timeseries"

    # Note: Based on _logger from (commit: 08723e0 - Oct 6, 2024), Windows logging formatting issue likely due to
    # due to a combination of only initializing module logger at function level and using hasHandlers() for child
    # loggers to determine propogation and handler assignment, which resulted in no module handler and propogation to
    # the root handler which defaults to standard error stream and has its own formatter
    if top_level is True:
        _USER_ROOT_HANDLER = logging.getLogger().hasHandlers()
        _USER_MODULE_HANDLERS[logger.name] = logger.handlers

    # Special case for parallel config to pass a user-defined logger specifically for parallel processing.
    if logger.name == parallel_module and not top_level and parallel_log_config:
        # Only QueueHandler will be in handler list
        logger.handlers.clear()
        if "queue" in parallel_log_config:
            logger.addHandler(QueueHandler(parallel_log_config["queue"]))
        if "level" in parallel_log_config:
            logger.setLevel(parallel_log_config["level"])

    if not logger.level:
        logger.setLevel(logging.INFO)

    # Check if user defined root handler or assigned a specific handler for module
    default_handlers = _USER_ROOT_HANDLER or _USER_MODULE_HANDLERS[logger.name]

    # Propagate to root if no user-defined module or a root handler is detected
    logger.propagate = False if _USER_MODULE_HANDLERS[logger.name] or not _USER_ROOT_HANDLER else True

    # Add or messages will repeat several times due to multiple handlers if same name used
    if not default_handlers and not (parallel_log_config or (logger.name == parallel_module and top_level)):
        # Safeguard; ensure a clean state for "extract_timeseries" since it is used in parallel and sequential contexts
        if logger.name == parallel_module:
            logger.handlers.clear()

        if flush:
            handler = _Flush(sys.stdout)
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger
