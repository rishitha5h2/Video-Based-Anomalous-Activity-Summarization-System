import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "vigil", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"vigil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
