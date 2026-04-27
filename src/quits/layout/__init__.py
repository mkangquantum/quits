"""Coordinate layout helpers for QLDPC code visualizations."""

from .base import Layout, LayoutMapping

__all__ = [
    "AbstractToricLayout",
    "BbToricLayout",
    "Layout",
    "LayoutMapping",
    "ToricLayout",
    "TransversalLayout",
]


def __getattr__(name):
    if name == "AbstractToricLayout":
        from .toric_common import AbstractToricLayout

        return AbstractToricLayout
    if name == "BbToricLayout":
        from .toric_bb import BbToricLayout

        return BbToricLayout
    if name == "ToricLayout":
        from .toric import ToricLayout

        return ToricLayout
    if name == "TransversalLayout":
        from .transversal import TransversalLayout

        return TransversalLayout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
