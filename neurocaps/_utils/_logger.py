"""Internal logging function and class for flushing"""

import logging, sys

class _Flush(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def _logger(name, flush=False):
    logger = logging.getLogger(name.split(".")[-1])

    if not logger.level: logger.setLevel(logging.INFO)

    # Works to see if root has handler and propagate if it does
    logger.propagate = logging.getLogger().hasHandlers()
    # Add or messages will repeat several times due to multiple handlers if same name used
    if not logger.hasHandlers():
        if flush: handler = _Flush(sys.stdout)
        else: handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger
