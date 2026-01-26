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

from .qlp import QlpPolyCode


class LscCode(QlpPolyCode):
    def __init__(self, lift_size, length):
        """
        :param lift_size: L, circulant lift size.
        :param length: l + 1, number of columns in the base matrix.
        """
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


__all__ = ["LscCode"]
