import logging

from pathlib import Path


__version__ = "0.0.0"


try:
    __version__ = open(Path(__file__).parent.parent / "VERSION.txt", "r").read()
except Exception:
    logger = logging.getLogger(__name__)
    logger.exception("No VERSION.txt found!")
