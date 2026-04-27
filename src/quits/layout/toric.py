"""Public toric layout factory."""

from __future__ import annotations

from .base import Layout

_TORIC_LAYOUT_ERROR = "ToricLayout only supports BbCode instances."


class ToricLayout:
    """Dispatch to the appropriate toric layout implementation for the code family."""

    def __new__(cls, code) -> Layout:
        from ..qldpc_code.bb import BbCode
        from .toric_bb import BbToricLayout

        if isinstance(code, BbCode):
            return BbToricLayout(code)
        raise ValueError(_TORIC_LAYOUT_ERROR)


__all__ = ["ToricLayout"]
