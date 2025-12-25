from __future__ import annotations
import logging
import sys
from dmc4d.config import CFG

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = getattr(logging, CFG.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger
