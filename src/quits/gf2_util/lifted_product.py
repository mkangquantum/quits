"""Circulant lifts, lifted product CSS codes over F2[G], and the mitten code family.

The lifted-product construction follows arXiv:2607.28795 and uses the same
block placement as that paper's reference implementation, so the matrices
built here match it.

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

from .linalg import gf2_rank, gf2_solve_square
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
# 2. The lifted product construction
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


# ─────────────────────────────────────────────────────────────────
# 3. Mitten codes
# ─────────────────────────────────────────────────────────────────


def is_valid_mitten(A: list, B: list, gd: MultTableGroup) -> bool:
    """Whether ``(A, B)`` satisfies Definition 2 of arXiv:2607.28795.

    Requires ``A = [a0 a1]`` and ``B = [b0 b1]`` of shape ``1 × 2`` over
    ``F2[G]``, every entry of weight 3, the identity present in ``a1`` and
    ``b1`` (the canonical form ``a1 = e + g4 + g5``), and ``L[a1]``, ``R[b1]``
    of full rank over GF(2) — the special case of the square invertibility
    condition that gives the canonical logical basis.
    """
    if len(A) != 1 or len(A[0]) != 2 or len(B) != 1 or len(B[0]) != 2:
        raise ValueError("mitten base matrices must have shape 1 x 2")
    (a0, a1), (b0, b1) = A[0], B[0]
    if any(len(x) != 3 for x in (a0, a1, b0, b1)):
        return False
    if gd.identity not in a1 or gd.identity not in b1:
        return False
    return (gf2_rank(left_rep(a1, gd)) == gd.n
            and gf2_rank(right_rep(b1, gd)) == gd.n)


def sample_mitten_base_matrices(gd: MultTableGroup, rng, max_trials: int = 10_000
                                ) -> tuple[list, list]:
    """Draw random weight-3 base matrices until they define a mitten code.

    Samples ``a0`` and ``b0`` as three distinct group elements, and ``a1``,
    ``b1`` as the identity plus two distinct non-identity elements, then keeps
    the first pair passing :func:`is_valid_mitten`.

    Args:
        gd: the group ``G``.
        rng: a ``numpy.random.Generator``.
        max_trials: how many draws to attempt before giving up.

    Returns:
        ``(A, B)`` as ``1 × 2`` ring matrices.

    Raises:
        RuntimeError: if no valid pair is found within ``max_trials``.
    """
    non_identity = np.arange(1, gd.n)

    def draw_free():
        return tuple(sorted(int(g) for g in
                            rng.choice(gd.n, size=3, replace=False)))

    def draw_pivot():
        return tuple(sorted([gd.identity] + [
            int(g) for g in rng.choice(non_identity, size=2, replace=False)
        ]))

    for _ in range(max_trials):
        A = [[draw_free(), draw_pivot()]]
        B = [[draw_free(), draw_pivot()]]
        if is_valid_mitten(A, B, gd):
            return A, B
    raise RuntimeError(
        f"no valid mitten base matrices found in {max_trials} trials"
    )


def mitten_canonical_logical_basis(A: list, B: list, gd: MultTableGroup
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Canonical logical basis of a mitten code (Theorem 1 of arXiv:2607.28795).

    For each ``g ∈ G``::

        X̄_g = (1_g, u_g, 0, 0, 0)ᵀ,      Z̄_g = (1_g, 0, v_g, 0, 0)ᵀ

    where ``u_g`` and ``v_g`` are the unique solutions of::

        R[b0]·1_g + R[b1]·u_g = 0,       L[a0]·1_g + L[a1]·v_g = 0.

    Uniqueness — hence the whole basis — comes from ``L[a1]`` and ``R[b1]``
    being full rank. Solving for all ``g`` at once means solving with the
    identity as right-hand side, so ``U = R[b1]⁻¹R[b0]`` has ``u_g`` as its
    ``g``-th column, and likewise ``V = L[a1]⁻¹L[a0]``.

    Returns:
        ``(Lx, Lz)``, each of shape ``(|G|, 5|G|)``, row ``g`` holding ``X̄_g``
        resp. ``Z̄_g``. They satisfy ``Lx · Lzᵀ = I``.

    Raises:
        ValueError: if ``(A, B)`` is not a valid mitten pair.
    """
    if not is_valid_mitten(A, B, gd):
        raise ValueError(
            "(A, B) is not a valid mitten pair — see is_valid_mitten"
        )
    (a0, a1), (b0, b1) = A[0], B[0]
    n = gd.n

    U = gf2_solve_square(right_rep(b1, gd), right_rep(b0, gd))
    V = gf2_solve_square(left_rep(a1, gd), left_rep(a0, gd))

    identity_block = np.eye(n, dtype=np.uint8)
    zero_block = np.zeros((n, n), dtype=np.uint8)

    # Row g of each basis; blocks are D1 … D5 left to right.
    Lx = np.concatenate(
        [identity_block, U.T, zero_block, zero_block, zero_block], axis=1
    )
    Lz = np.concatenate(
        [identity_block, zero_block, V.T, zero_block, zero_block], axis=1
    )
    return Lx.astype(np.uint8), Lz.astype(np.uint8)


def conjugate_pair_overlaps(Lx: np.ndarray, Lz: np.ndarray) -> np.ndarray:
    """Support-overlap sizes ``|supp(Lx_i) ∩ supp(Lz_i)|`` for each ``i``."""
    return np.sum((np.asarray(Lx) != 0) & (np.asarray(Lz) != 0), axis=1)
