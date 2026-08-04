"""CSS logical operators, code parameters, and classical code distance.

Input normalization: compute_lz, compute_lz_and_lx, compute_code_distance, and
verify_css_logicals reduce entries mod 2 via _as_gf2 (`& 1`), while the
lifted-product diagnostics (check_css, compute_k, check_weights, qubit_weights)
treat any nonzero entry as 1 (`!= 0`). The two conventions agree on {0,1}
inputs.
"""

import numpy as np
from typing import Dict, Tuple

from .linalg import (
    _as_gf2,
    _gf2_inv_square,
    gf2_nullspace_basis,
    gf2_rank,
    gf2_row_basis,
    gf2_row_span,
    gf2_rref,
)


# ============================================================
# CSS diagnostics and code parameters
# ============================================================

def check_css(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    """CSS orthogonality: ``Hx · Hzᵀ == 0`` (mod 2)."""
    Hx01 = (np.asarray(Hx) != 0).astype(np.uint8)
    Hz01 = (np.asarray(Hz) != 0).astype(np.uint8)
    return bool(np.all((Hx01.astype(np.int64) @ Hz01.T.astype(np.int64)) % 2 == 0))


def compute_k(Hx: np.ndarray, Hz: np.ndarray) -> int:
    """Number of logical qubits ``k = n - rank(Hx) - rank(Hz)`` over GF(2).

    Only meaningful when ``Hx · Hzᵀ = 0``; check that with :func:`check_css`.
    """
    Hx01 = (np.asarray(Hx) != 0).astype(np.uint8)
    Hz01 = (np.asarray(Hz) != 0).astype(np.uint8)
    return int(Hx01.shape[1] - gf2_rank(Hx01) - gf2_rank(Hz01))


def check_weights(Hx: np.ndarray, Hz: np.ndarray) -> dict:
    """Maximum row weight of ``Hx`` and ``Hz`` — the code's check weights.

    The check weight is how many data qubits each check acts on.
    """
    return {
        "Hx_check_weight": int(np.max(np.sum(Hx != 0, axis=1))),
        "Hz_check_weight": int(np.max(np.sum(Hz != 0, axis=1))),
    }


def qubit_weights(Hx: np.ndarray, Hz: np.ndarray) -> dict:
    """Maximum column weight of ``Hx`` and ``Hz`` — the code's qubit weights.

    The qubit weight is the other side of the Tanner graph: how many checks
    each data qubit is connected to. ``qubit_weight`` is the total degree,
    maximised over qubits, which can be smaller than the sum of the two
    per-type maxima when they are attained on different qubits.
    """
    cols_x = np.sum(np.asarray(Hx) != 0, axis=0)
    cols_z = np.sum(np.asarray(Hz) != 0, axis=0)
    return {
        "Hx_qubit_weight": int(np.max(cols_x)),
        "Hz_qubit_weight": int(np.max(cols_z)),
        "qubit_weight": int(np.max(cols_x + cols_z)),
    }


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
    if not check_css(hx, hz):
        raise ValueError("Not a CSS pair: hx @ hz.T != 0 (mod 2)")

    n = hx.shape[1]
    k = compute_k(hx, hz)
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
