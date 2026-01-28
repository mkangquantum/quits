# This code reimplements and adapts several GF(2) linear-algebra and CSS
# logical-operator routines from the QuantumGizmos ldpc package
# (https://github.com/quantumgizmos/ldpc), in particular the file
#   - src_python/ldpc/mod2/mod2_numpy.py


import numpy as np
from typing import Tuple, Dict


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


def in_rowspace(v: np.ndarray, H: np.ndarray) -> bool:
    """
    Check if v is in rowspace(H) over GF(2) by solving H^T a = v.
    """
    v = _as_gf2(v).reshape(-1)
    H = _as_gf2(H)
    return gf2_solve(H.T, v) is not None


# ============================================================
# CSS logicals + classical code's distance 
# ============================================================

def compute_lz(hz: np.ndarray, hx: np.ndarray) -> np.ndarray:
    """
    Compute logical Z operators for CSS code defined by hx, hz:

      ker(hx) \\ rowspace(hz)

    Implemented by stacking [row_basis(hz); nullspace(hx)] and selecting the
    kernel rows whose *row indices* are pivot columns of RREF(stack.T).
    """
    ker_hx = gf2_nullspace_basis(hx)   # rows: basis for ker(hx)
    im_hz  = gf2_row_basis(hz)         # rows: basis for rowspace(hz)

    log_stack = np.vstack([im_hz, ker_hx]).astype(np.uint8, copy=False)

    # Pivots of stack.T: columns correspond to rows of stack
    _, piv = gf2_rref(log_stack.T)
    piv = set(piv.tolist())

    image_rank = im_hz.shape[0]
    keep = [i for i in range(image_rank, log_stack.shape[0]) if i in piv]
    return log_stack[keep, :]


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


def compute_lz_and_lx(hz: np.ndarray, hx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (lz, lx) where:
      - lz is computed by compute_lz
      - lx is then *dualized* so that (lz @ lx.T) % 2 == I_k,
        with lx chosen from ker(hz).

    This guarantees correct pairing whenever (hx,hz) define a valid CSS code.
    """
    hz = _as_gf2(hz)
    hx = _as_gf2(hx)

    if hz.ndim != 2 or hx.ndim != 2:
        raise ValueError("hz and hx must be 2D arrays")
    if hz.shape[1] != hx.shape[1]:
        raise ValueError("hz and hx must have the same number of columns")
    if np.any((hx @ hz.T) & 1):
        raise ValueError("Not a CSS pair: hx @ hz.T != 0 (mod 2)")

    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k < 0:
        raise ValueError(f"Computed k={k}<0. Checks inconsistent?")
    if k == 0:
        return np.zeros((0, n), dtype=np.uint8), np.zeros((0, n), dtype=np.uint8)

    # 1) Obtain LZ
    lz = compute_lz(hz, hx)
    if lz.shape[0] != k:
        raise RuntimeError(f"compute_lz returned {lz.shape[0]} logical Zs, expected k={k}")

    # 2) Build LX ⊂ ker(hz) such that LX LZ^T = I
    ker_hz = gf2_nullspace_basis(hz)  # rows span ker(hz), dim = n - rank(hz)
    W = (ker_hz @ lz.T) & 1           # (dim ker_hz, k)

    if gf2_rank(W) < k:
        raise RuntimeError(
            "Cannot dualize: ker(hz) does not contain enough vectors with "
            "independent commutation against the chosen lz."
        )

    # Pick k independent rows of W (and corresponding kernel vectors)
    chosen = []
    piv = {}  # pivot col -> vector (as in gf2_row_basis style)
    for i in range(W.shape[0]):
        v = W[i].copy()
        while True:
            ones = np.flatnonzero(v)
            if ones.size == 0:
                break
            c = int(ones[0])
            if c in piv:
                v ^= piv[c]
            else:
                piv[c] = v
                chosen.append(i)
                break
        if len(chosen) == k:
            break

    chosen = np.array(chosen, dtype=int)
    W_sel = W[chosen, :]          # (k,k) full rank
    K_sel = ker_hz[chosen, :]     # (k,n)

    # Want LX so that (LX @ LZ^T)=I.
    # If we set LX = (W_sel^{-1}) @ K_sel then:
    #   LX LZ^T = (W_sel^{-1} K_sel) LZ^T = W_sel^{-1} (K_sel LZ^T) = W_sel^{-1} W_sel = I
    W_inv = _gf2_inv_square(W_sel)
    lx = (W_inv @ K_sel) & 1

    # sanity
    if not np.array_equal((lz @ lx.T) & 1, np.eye(k, dtype=np.uint8)):
        raise RuntimeError("Internal error: dualization failed to produce lz @ lx.T = I.")

    return lz.astype(np.uint8), lx.astype(np.uint8)


def compute_code_distance(H: np.ndarray):
    """
    Compute code distance from parity check matrix H over GF(2):
    """
    ker = gf2_nullspace_basis(H)
    if ker.shape[0] == 0:
        return np.inf
    cw = gf2_row_span(ker)
    return int(np.min(np.sum(cw, axis=1)))

def verify_css_logicals(
    hz: np.ndarray,
    hx: np.ndarray,
    lz: np.ndarray,
    lx: np.ndarray,
) -> Dict[str, object]:
    """
    Verify that user-supplied CSS logical operators (lz, lx) are valid and complete
    for the CSS stabilizers (hz, hx), in the same spirit as verify_fixed_logical_Xs.

    Assumes gf2_util.py provides:
      - _as_gf2
      - gf2_rank
      - gf2_nullspace_basis

    Returns a dict with booleans + ranks + dims + final 'all_tests_passed'.
    """
    hz = _as_gf2(hz)
    hx = _as_gf2(hx)
    lz = _as_gf2(lz)
    lx = _as_gf2(lx)

    n = hz.shape[1]
    if hx.shape[1] != n or lx.shape[1] != n or lz.shape[1] != n:
        raise ValueError("hz, hx, lz, lx must all have the same number of columns n")

    report: Dict[str, object] = {}

    # --- CSS commutation ---
    css_ok = not np.any((hx @ hz.T) & 1)
    report["css_condition"] = css_ok

    # --- Logical commutation with opposite stabilizers ---
    lz_commutes_with_X = not np.any((hx @ lz.T) & 1)
    lx_commutes_with_Z = not np.any((hz @ lx.T) & 1)
    report["lz_commutes_with_X"] = lz_commutes_with_X
    report["lx_commutes_with_Z"] = lx_commutes_with_Z

    # --- Ranks / expected k ---
    rank_hz = gf2_rank(hz)
    rank_hx = gf2_rank(hx)
    rank_lz = gf2_rank(lz)
    rank_lx = gf2_rank(lx)

    report["rank_hz"] = int(rank_hz)
    report["rank_hx"] = int(rank_hx)
    report["rank_lz"] = int(rank_lz)
    report["rank_lx"] = int(rank_lx)

    k_expected = int(n - rank_hx - rank_hz)
    report["k_expected"] = k_expected

    # --- Independence modulo stabilizers ---
    rank_hz_lz = gf2_rank(np.vstack([hz, lz]))
    rank_hx_lx = gf2_rank(np.vstack([hx, lx]))
    lz_indep_mod_stab = (rank_hz_lz == rank_hz + rank_lz)
    lx_indep_mod_stab = (rank_hx_lx == rank_hx + rank_lx)

    report["lz_independent_mod_Z_stabilizers"] = lz_indep_mod_stab
    report["lx_independent_mod_X_stabilizers"] = lx_indep_mod_stab
    report["rank_hz_plus_lz"] = int(rank_hz_lz)
    report["rank_hx_plus_lx"] = int(rank_hx_lx)

    # --- Spanning normalizers (ker spaces) ---
    ker_hz = gf2_nullspace_basis(hz)
    ker_hx = gf2_nullspace_basis(hx)
    dim_ker_hz = int(ker_hz.shape[0])  # = n - rank(hz)
    dim_ker_hx = int(ker_hx.shape[0])  # = n - rank(hx)

    report["dim_ker_hz"] = dim_ker_hz
    report["dim_ker_hx"] = dim_ker_hx

    spans_ker_hz = (rank_hx_lx == dim_ker_hz)
    spans_ker_hx = (rank_hz_lz == dim_ker_hx)
    report["hx_plus_lx_spans_ker_hz"] = spans_ker_hz
    report["hz_plus_lz_spans_ker_hx"] = spans_ker_hx

    # --- Pairing matrix ---
    overlap_weight = lx @ lz.T
    pairing = overlap_weight % 2
    is_square_pairing = pairing.shape[0] == pairing.shape[1]
    same_logicals_ZX_anticommute = (
        is_square_pairing
        and np.array_equal(np.diag(pairing), np.ones(pairing.shape[0], dtype=np.uint8))
    )
    off_diag = pairing.copy()
    if is_square_pairing and pairing.shape[0] > 0:
        np.fill_diagonal(off_diag, 0)
    different_logicals_ZX_commute = is_square_pairing and not np.any(off_diag)
    report["same_logicals_ZX_anticommute"] = same_logicals_ZX_anticommute
    report["different_logicals_ZX_commute"] = different_logicals_ZX_commute

    # --- Final verdict ---
    ok = (
        css_ok
        and lz_commutes_with_X
        and lx_commutes_with_Z
        and (rank_lz == k_expected)
        and (rank_lx == k_expected)
        and lz_indep_mod_stab
        and lx_indep_mod_stab
        and spans_ker_hz
        and spans_ker_hx
        and same_logicals_ZX_anticommute
        and different_logicals_ZX_commute
    )
    report["all_tests_passed"] = bool(ok)

    return report
