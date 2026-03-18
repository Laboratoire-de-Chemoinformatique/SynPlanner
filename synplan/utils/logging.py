"""
Generic logging helpers for scripts and notebooks.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from IPython import get_ipython

# --------------------------------------------------------------------------- #
#                               Helper classes                                #
# --------------------------------------------------------------------------- #


class DisableLogger:
    """Context‑manager that suppresses *all* logging inside its scope."""

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)


class HiddenPrints:
    """Context‑manager that suppresses *print* output inside its scope."""

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._orig


# --------------------------------------------------------------------------- #
#                         Notebook‑aware console handler                      #
# --------------------------------------------------------------------------- #


def _in_notebook() -> bool:
    ip = get_ipython()
    return bool(ip) and ip.__class__.__name__ == "ZMQInteractiveShell"


class TqdmHandler(logging.StreamHandler):
    """Write via tqdm.write so log lines don't break progress bars."""

    def emit(self, record):
        try:
            from tqdm.auto import tqdm

            tqdm.write(self.format(record), end=self.terminator)
        except ModuleNotFoundError:
            super().emit(record)


# --------------------------------------------------------------------------- #
#                           Public initialisation API                         #
# --------------------------------------------------------------------------- #


def init_logger(
    *,
    name: str = "app",
    console_level: str | int = "ERROR",
    file_level: str | int = "INFO",
    log_dir: str | os.PathLike = ".",
    redirect_tqdm: bool = True,
) -> tuple[logging.Logger, str]:
    """
    Initialise (or fetch) a namespaced logger that works in scripts &
    notebooks.  Idempotent ‑ safe to call multiple times.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel("DEBUG")  # capture everything; handlers filter

    # console / notebook handler
    if _in_notebook() or (redirect_tqdm and "tqdm" in sys.modules):
        ch: logging.Handler = TqdmHandler()
    else:
        ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # rotating file handler (one file per session)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(Path(log_dir) / f"{name}_{stamp}.log", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # logger.propagate = False # Removed correctly
    log_file_path = fh.baseFilename
    logger.info("Logging initialised → %s", log_file_path)
    return logger, log_file_path  # <-- Return path too


# --------------------------------------------------------------------------- #
#                 Optional Ray‑specific configuration helpers                 #
# --------------------------------------------------------------------------- #


def silence_logger(
    logger_name: str,
    level: int | str = logging.ERROR,
):
    """
    Raise the threshold of a chatty library logger.
    Useful inside worker process initializers.
    """
    logging.getLogger(logger_name).setLevel(level)
