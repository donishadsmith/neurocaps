"""Internal logging function and class for flushing"""

import logging, sys

# Global variables to determine if a handler is user defined or defined by OS
_USER_ROOT_HANDLER = None
_USER_MODULE_HANDLERS = {}

class _Flush(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def _logger(name, flush=False, top_level=True):
    global _USER_ROOT_HANDLER, _USER_MODULE_HANDLERS

    logger = logging.getLogger(name.split(".")[-1])

    # Windows appears to assign stderr has the root handler after the top level loggers are assigned, which causes
    # any loggers not assigned at top level to adopt this handler. Global variable used to assess if the base root
    # handler is user defined or assigned by the system
    if top_level == True:
        _USER_ROOT_HANDLER = logging.getLogger().hasHandlers()
        _USER_MODULE_HANDLERS[logger.name] = logging.getLogger(logger.name).hasHandlers()

    if not logger.level: logger.setLevel(logging.INFO)

    # Check if user defined root handler or assigned a specific handler for module
    default_handlers = _USER_ROOT_HANDLER or _USER_MODULE_HANDLERS[logger.name]

    # Works to see if root has handler and propagate if it does
    logger.propagate = True if _USER_ROOT_HANDLER else False

    # Add or messages will repeat several times due to multiple handlers if same name used
    if not default_handlers and not (logger.name == "_extract_timeseries" and top_level):
        # If no user specified default handler, any handler is assigned by OS and is cleared
        if logger.name == "_extract_timeseries": logger.handlers.clear()

        if flush: handler = _Flush(sys.stdout)
        else: handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger
