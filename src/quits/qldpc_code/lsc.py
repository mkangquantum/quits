"""
Lift-connected surface code (LCS) construction.

We parameterize the code by:
- length = l + 1 (number of columns in the base matrix)
- lift_size = L (circulant lift size)

The base matrix has shape (l, l + 1) and uses polynomial entries:
    I  and  I + P
on the two diagonals.
"""

import numpy as np

from ..noise import ErrorModel
from .circuit_construction.circuit_build_options import CircuitBuildOptions
from .qlp import QlpPolyCode


class LscCode(QlpPolyCode):
    def __init__(self, lift_size, length):
        """
        :param lift_size: L, circulant lift size.
        :param length: l + 1, number of columns in the base matrix.
        """
        # Reference: J. Old, M. Rispler, and M. Muller, arXiv:2401.02911 (lift-connected surface codes).
        if length < 2:
            raise ValueError("length must be at least 2 so that l = length - 1 is positive.")

        l = length - 1
        b = [[[] for _ in range(length)] for _ in range(l)]
        for i in range(l):
            b[i][i] = [0]
            b[i][i + 1] = [0, 1]

        self.length = length
        self.l = l
        self.lift_size = lift_size
        self.b = np.array(b, dtype=object)

        super().__init__(b, b, lift_size)

    def build_circuit(
        self,
        strategy="cardinal",
        seed=1,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        **opts,
    ):
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")
        return super().build_circuit(
            strategy=strategy,
            seed=seed,
            error_model=error_model,
            num_rounds=num_rounds,
            basis=basis,
            circuit_build_options=circuit_build_options,
            **opts,
        )


__all__ = ["LscCode"]
