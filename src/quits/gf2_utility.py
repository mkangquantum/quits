import numpy as np

def gf2_rref_pivots(H: np.ndarray):
    """Return pivot columns of H in RREF over GF(2)."""
    A = (np.asarray(H) & 1).astype(np.uint8, copy=True)
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
        # eliminate column c in all other rows
        ones = np.where(A[:, c] == 1)[0]
        ones = ones[ones != r]
        if ones.size:
            A[ones, :] ^= A[r, :]
        pivots.append(c)
        r += 1
    return np.array(pivots, dtype=int)

def gf2_coset_reps_rowspace(H: np.ndarray) -> np.ndarray:
    """
    Canonical reps for F2^n / rowspace(H):
    pick standard basis e_j for the non-pivot columns of RREF(H).
    """
    H = (np.asarray(H) & 1).astype(np.uint8, copy=False)
    n = H.shape[1]
    piv = set(gf2_rref_pivots(H).tolist())
    nonpiv = [c for c in range(n) if c not in piv]  # size = n - rank(H) = k (if full row rank)
    E = np.zeros((len(nonpiv), n), dtype=np.uint8)
    for t, c in enumerate(nonpiv):
        E[t, c] = 1
    return E

def gf2_rank(H: np.ndarray) -> int:
    return len(gf2_rref_pivots(H))

def in_rowspace(v: np.ndarray, H: np.ndarray) -> bool:
    """Check if v is in rowspace(H) over GF(2) by solving H^T a = v."""
    H = (np.asarray(H) & 1).astype(np.uint8, copy=False)
    v = (np.asarray(v).reshape(-1) & 1).astype(np.uint8, copy=False)
    m, n = H.shape
    # Solve H^T a = v for a in F2^m
    A = H.T.copy()
    b = v.copy()

    # Gaussian elimination
    Aug = np.concatenate([A, b[:, None]], axis=1)
    r = 0
    pivcol = -np.ones(n, dtype=int)
    for c in range(m):
        if r >= n:
            break
        rows = np.where(Aug[r:, c] == 1)[0]
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        if p != r:
            Aug[[r, p]] = Aug[[p, r]]
        pivcol[r] = c
        ones = np.where(Aug[:, c] == 1)[0]
        ones = ones[ones != r]
        if ones.size:
            Aug[ones, c:] ^= Aug[r, c:]
        r += 1

    # infeasible row: 0...0 | 1
    if np.any(np.all(Aug[:, :m] == 0, axis=1) & (Aug[:, m] == 1)):
        return False
    return True