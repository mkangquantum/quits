# This code reimplements and adapts several GF(2) linear-algebra and CSS
# logical-operator routines from the QuantumGizmos ldpc package
# (https://github.com/quantumgizmos/ldpc), in particular the file
#   - src_python/ldpc/mod2/mod2_numpy.py


import numpy as np
from typing import Tuple


# ============================================================
# Minimal GF(2) linear algebra utilities (numpy)
# ============================================================

def _as_gf2(A) -> np.ndarray:
    """Convert input to uint8 array reduced mod 2."""
    return (np.asarray(A) & 1).astype(np.uint8, copy=False)


def gf2_rref(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduced row echelon form over GF(2).

    Returns:
      R: RREF(H) as uint8
      pivots: pivot column indices as int array
    """
    A = _as_gf2(H).copy()
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        rows = np.where(A[r:, c] == 1)[0]
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        if p != r:
            A[[r, p], :] = A[[p, r], :]
        # eliminate pivot column in all other rows (RREF)
        ones = np.where(A[:, c] == 1)[0]
        ones = ones[ones != r]
        if ones.size:
            A[ones, :] ^= A[r, :]
        pivots.append(c)
        r += 1
    return A, np.array(pivots, dtype=int)


def gf2_rank(H: np.ndarray) -> int:
    """Rank over GF(2)."""
    return int(gf2_rref(H)[1].size)


def gf2_row_basis(H: np.ndarray) -> np.ndarray:
    """
    Row basis for rowspace(H) over GF(2).
    Returned as nonzero rows of RREF(H).
    """
    R, _ = gf2_rref(H)
    nz = np.where(np.any(R == 1, axis=1))[0]
    return R[nz, :].astype(np.uint8, copy=False)


def gf2_nullspace_basis(H: np.ndarray) -> np.ndarray:
    """
    Basis for the (right) nullspace of H over GF(2): {x : H x = 0}.
    Returned as rows. Shape: (nullity, n).
    """
    H = _as_gf2(H)
    if H.ndim != 2:
        raise ValueError("H must be 2D")
    _, n = H.shape

    R, pivots = gf2_rref(H)
    pivset = set(pivots.tolist())
    free = [c for c in range(n) if c not in pivset]

    if not free:
        return np.zeros((0, n), dtype=np.uint8)

    rank = pivots.size
    basis = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        # Since R is RREF: for pivot row i with pivot column pivots[i],
        # equation is x[pivot] + sum_{c in free} R[i,c] x[c] = 0
        # => x[pivot] = sum_{c in free} R[i,c] x[c]
        for i in range(rank):
            pc = pivots[i]
            s = np.uint8(0)
            for c in free:
                if R[i, c]:
                    s ^= x[c]
            x[pc] = s
        basis.append(x)

    return np.stack(basis, axis=0)


def gf2_coset_reps_rowspace(H: np.ndarray) -> np.ndarray:
    """
    Canonical reps for F2^n / rowspace(H):
    pick standard basis e_j for non-pivot columns of RREF(H).
    """
    H = _as_gf2(H)
    n = H.shape[1]
    piv = set(gf2_rref(H)[1].tolist())
    nonpiv = [c for c in range(n) if c not in piv]
    E = np.zeros((len(nonpiv), n), dtype=np.uint8)
    for t, c in enumerate(nonpiv):
        E[t, c] = 1
    return E


def gf2_row_span(V: np.ndarray) -> np.ndarray:
    """
    Enumerate all NONZERO linear combinations of the rows of V over GF(2).
    WARNING: exponential in k = number of rows. Returns 2^k - 1 vectors.
    """
    V = _as_gf2(V)
    k, n = V.shape
    if k == 0:
        return np.zeros((0, n), dtype=np.uint8)

    out = np.zeros((2**k - 1, n), dtype=np.uint8)

    # Gray-code accumulation: successive masks differ by one bit -> one XOR update
    acc = np.zeros(n, dtype=np.uint8)
    prev_gray = 0
    idx = 0
    for mask in range(1, 2**k):
        gray = mask ^ (mask >> 1)
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        acc ^= V[bit]
        out[idx] = acc
        idx += 1
        prev_gray = gray

    return out


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """
    Solve A x = b over GF(2). Return one solution x or None if infeasible.

    A: (m,n), b: (m,)
    """
    A = _as_gf2(A).copy()
    b = _as_gf2(b).reshape(-1).copy()

    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("dimension mismatch: b must have length m")

    Aug = np.concatenate([A, b[:, None]], axis=1)

    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        rows = np.where(Aug[r:, c] == 1)[0]
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        if p != r:
            Aug[[r, p], :] = Aug[[p, r], :]
        ones = np.where(Aug[:, c] == 1)[0]
        ones = ones[ones != r]
        if ones.size:
            Aug[ones, c:] ^= Aug[r, c:]
        pivot_cols.append(c)
        r += 1

    # infeasible row: 0...0 | 1
    if np.any(np.all(Aug[:, :n] == 0, axis=1) & (Aug[:, n] == 1)):
        return None

    x = np.zeros(n, dtype=np.uint8)
    for rr, pc in enumerate(pivot_cols):
        x[pc] = Aug[rr, n]
    return x


def gf2_solve_square(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A @ X = B over GF(2) for a square invertible A and matrix B.

    Complements :func:`gf2_solve` (rectangular A, vector b, particular
    solution or None): here the system has a unique solution, computed as
    A^{-1} B mod 2.

    Args:
        A: (n, n) square binary matrix (np.ndarray), invertible over GF(2).
        B: (n, m) binary right-hand side (np.ndarray).

    Returns:
        X: the unique (n, m) uint8 solution (np.ndarray).

    Raises:
        ValueError: if A is not square, is shape-incompatible with B,
            or is not full rank over GF(2).
    """
    A = _as_gf2(A)
    B = _as_gf2(B)
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError(f"A must be square; got shape {A.shape}")
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B must have shape ({n}, m); got {B.shape}")
    try:
        A_inv = _gf2_inv_square(A)
    except RuntimeError:
        raise ValueError("A is not full rank over GF(2)")
    return ((A_inv.astype(np.int64) @ B.astype(np.int64)) & 1).astype(np.uint8)


def in_rowspace(v: np.ndarray, H: np.ndarray) -> bool:
    """
    Check if v is in rowspace(H) over GF(2) by solving H^T a = v.
    """
    v = _as_gf2(v).reshape(-1)
    H = _as_gf2(H)
    return gf2_solve(H.T, v) is not None


def _gf2_inv_square(A: np.ndarray) -> np.ndarray:
    """Inverse of a full-rank square matrix over GF(2) via Gauss-Jordan."""
    A = (np.asarray(A) & 1).astype(np.uint8, copy=True)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square")
    Aug = np.concatenate([A, np.eye(n, dtype=np.uint8)], axis=1)

    r = 0
    for c in range(n):
        rows = np.where(Aug[r:, c] == 1)[0]
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        if p != r:
            Aug[[r, p], :] = Aug[[p, r], :]
        ones = np.where(Aug[:, c] == 1)[0]
        ones = ones[ones != r]
        if ones.size:
            Aug[ones, :] ^= Aug[r, :]
        r += 1
        if r == n:
            break

    if not np.array_equal(Aug[:, :n], np.eye(n, dtype=np.uint8)):
        raise RuntimeError("Matrix not invertible over GF(2).")
    return Aug[:, n:]
