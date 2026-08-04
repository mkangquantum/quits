"""Circulant lifts and lifted product CSS codes over F2[G].

The lifted-product construction follows arXiv:2607.28795 and uses the same
block placement as that paper's reference implementation, so the matrices
built here match it. The mitten code family built on this construction lives
in :mod:`quits.qldpc_code.mitten`.

Block placement of the lifted product (``⊗_lp`` is the block-Kronecker over
the group algebra), for ``A`` of shape ``ma × na`` and ``B`` of ``mb × nb``::

    Hx = [ L[A] ⊗_lp id_nb  |  id_ma ⊗_lp R[B*] ]
    Hz = [ id_na ⊗_lp R[B]  |  L[A*] ⊗_lp id_mb ]

Column blocks are ordered ``(ja·nb + ib)`` in the left section and
``(ia·mb + jb)`` in the right section, each block spanning ``|G|`` columns.
For the mitten case ``ma = mb = 1``, ``na = nb = 2`` this gives the five data
blocks ``D1 … D5`` of the paper's Eq. (2) in that order.

Generic CSS diagnostics (``check_css``, ``compute_k``, ``check_weights``,
``qubit_weights``, ``verify_css_logicals``) live in :mod:`quits.gf2_util.css`.
"""

import numpy as np
from scipy.linalg import circulant

from .group import MultTableGroup, dagger, left_rep, right_rep


# ─────────────────────────────────────────────────────────────────
# 1. Circulant lifts — quasi-cyclic codes
# ─────────────────────────────────────────────────────────────────
#
# The lift over a cyclic group in matrix form: get_circulant_mat(size, power)
# equals left_rep((power % size,), cyclic(size)) but returns dtype int, and
# lift_enc accumulates repeated powers instead of cancelling them mod 2.


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


# ─────────────────────────────────────────────────────────────────
# 2. Lifted product over an arbitrary group
# ─────────────────────────────────────────────────────────────────


def _check_rectangular(M: list, name: str) -> tuple[int, int]:
    rows = len(M)
    cols = len(M[0])
    if not all(len(row) == cols for row in M):
        raise ValueError(
            f"{name} is not rectangular: row lengths "
            f"{[len(r) for r in M]} (expected all == {cols})."
        )
    return rows, cols


def build_Hx_from_lifted_product(A: list, B: list, gd: MultTableGroup) -> np.ndarray:
    """X parity-check matrix of ``LP(A, B)``.

    ``Hx = [ L[A] ⊗_lp id_nb | id_ma ⊗_lp R[B*] ]``, of shape
    ``(ma·nb·|G|) × (na·nb + ma·mb)·|G|``.
    """
    ma, na = _check_rectangular(A, "A")
    mb, nb = _check_rectangular(B, "B")
    n = gd.n

    Hx = np.zeros((ma * nb * n, (na * nb + ma * mb) * n), dtype=np.uint8)

    # Left section: L[A] ⊗_lp id_nb
    for ia in range(ma):
        for ja in range(na):
            LA = left_rep(A[ia][ja], gd)
            for ib in range(nb):
                r = (ia * nb + ib) * n
                c = (ja * nb + ib) * n
                Hx[r:r + n, c:c + n] ^= LA

    # Right section: id_ma ⊗_lp R[B*],  with B*[ib][jb] = B[jb][ib]*
    left_cols = na * nb * n
    for ib in range(nb):
        for jb in range(mb):
            RBd = right_rep(dagger(B[jb][ib], gd), gd)
            for ia in range(ma):
                r = (ia * nb + ib) * n
                c = left_cols + (ia * mb + jb) * n
                Hx[r:r + n, c:c + n] ^= RBd

    return Hx


def build_Hz_from_lifted_product(A: list, B: list, gd: MultTableGroup) -> np.ndarray:
    """Z parity-check matrix of ``LP(A, B)``.

    ``Hz = [ id_na ⊗_lp R[B] | L[A*] ⊗_lp id_mb ]``, of shape
    ``(na·mb·|G|) × (na·nb + ma·mb)·|G|``.
    """
    ma, na = _check_rectangular(A, "A")
    mb, nb = _check_rectangular(B, "B")
    n = gd.n

    Hz = np.zeros((na * mb * n, (na * nb + ma * mb) * n), dtype=np.uint8)

    # Left section: id_na ⊗_lp R[B]
    for ib in range(mb):
        for jb in range(nb):
            RB = right_rep(B[ib][jb], gd)
            for ja in range(na):
                r = (ja * mb + ib) * n
                c = (ja * nb + jb) * n
                Hz[r:r + n, c:c + n] ^= RB

    # Right section: L[A*] ⊗_lp id_mb,  with A*[ja][ia] = A[ia][ja]*
    left_cols = na * nb * n
    for ja in range(na):
        for ia in range(ma):
            LAd = left_rep(dagger(A[ia][ja], gd), gd)
            for ib in range(mb):
                r = (ja * mb + ib) * n
                c = left_cols + (ia * mb + ib) * n
                Hz[r:r + n, c:c + n] ^= LAd

    return Hz
