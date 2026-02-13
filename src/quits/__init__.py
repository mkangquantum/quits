"""QUITS package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quits")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

from .api import *

