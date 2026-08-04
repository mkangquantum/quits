"""Deprecated location for the circulant-lift helpers.

The implementations moved to :mod:`quits.gf2_util.lifted_product`; import
``get_circulant_mat``, ``lift``, and ``lift_enc`` from ``quits.gf2_util``
instead. This module re-exports them so existing imports keep working.
"""

from ..gf2_util.lifted_product import get_circulant_mat, lift, lift_enc

__all__ = ["get_circulant_mat", "lift", "lift_enc"]
