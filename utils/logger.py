import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            Path(log_dir) / f"atlas-{datetime.now().strftime('%Y-%m')}.log",
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
