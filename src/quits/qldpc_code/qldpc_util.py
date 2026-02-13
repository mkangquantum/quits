"""Matrix helper utilities for QLDPC code construction."""

import numpy as np
from scipy.linalg import circulant


def get_circulant_mat(size: int, power: int) -> np.ndarray:
    """Return the size x size circulant shift matrix for the given power."""
    return circulant(np.eye(size, dtype=int)[:, power])


def lift(lift_size: int, h_base: np.ndarray, h_base_placeholder: np.ndarray) -> np.ndarray:
    """
    Lift a monomial base matrix into a binary parity-check matrix.

    :param lift_size: Size of cyclic matrix to which each monomial entry is lifted.
    :param h_base: Base matrix where each entry is the power of the monomial.
    :param h_base_placeholder: Placeholder matrix where each non-zero entry of the base matrix is replaced by 1.
    :return: Lifted matrix.
    """
    h = np.zeros((h_base.shape[0] * lift_size, h_base.shape[1] * lift_size), dtype=int)
    for i in range(h_base.shape[0]):
        for j in range(h_base.shape[1]):
            if h_base_placeholder[i, j] != 0:
                h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] = get_circulant_mat(
                    lift_size, h_base[i, j]
                )
    return h


def lift_enc(lift_size: int, h_base_enc: np.ndarray, h_base_placeholder: np.ndarray) -> np.ndarray:
    """
    Lift an encoded polynomial base matrix into a binary parity-check matrix.

    :param lift_size: Size of cyclic matrix to which each polynomial term is lifted.
    :param h_base_enc: Base matrix where each entry ENCODEs powers of polynomial terms in base ``lift_size``.
    :param h_base_placeholder: Placeholder matrix where each non-zero entry of the base matrix is replaced by 1.
    :return: Lifted matrix.
    """
    h = np.zeros((h_base_enc.shape[0] * lift_size, h_base_enc.shape[1] * lift_size), dtype=int)
    for i in range(h_base_enc.shape[0]):
        for j in range(h_base_enc.shape[1]):
            if h_base_placeholder[i, j] != 0:
                hij_enc = h_base_enc[i, j]
                if hij_enc == 0:
                    h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] = get_circulant_mat(
                        lift_size, 0
                    )
                else:
                    while hij_enc > 0:
                        power = hij_enc % lift_size
                        h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] += get_circulant_mat(
                            lift_size, power
                        )
                        hij_enc = hij_enc // lift_size
    return h
