"""Unit tests for quits.gf2_util.linalg (GF(2) linear algebra core)."""

import numpy as np
import pytest

from quits.gf2_util import (
    _as_gf2,
    _gf2_inv_square,
    gf2_coset_reps_rowspace,
    gf2_nullspace_basis,
    gf2_rank,
    gf2_row_basis,
    gf2_row_span,
    gf2_rref,
    gf2_solve,
    gf2_solve_square,
    in_rowspace,
)


def _random_binary(shape, seed):
    return np.random.default_rng(seed).integers(0, 2, size=shape, dtype=np.uint8)


def _random_invertible(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if gf2_rank(A) == n:
            return A


def test_as_gf2_mod2_normalization():
    A = np.array([[0, 1, 2], [3, -1, 4]])
    out = _as_gf2(A)
    # & 1 semantics: even entries -> 0, odd entries -> 1 (including -1)
    assert np.array_equal(out, np.array([[0, 1, 0], [1, 1, 0]]))
    assert out.dtype == np.uint8


def test_rref_known_matrix_and_pivot_dtype():
    M = np.array([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    R, pivots = gf2_rref(M)
    assert isinstance(pivots, np.ndarray)
    assert pivots.dtype == np.dtype(int)
    assert np.array_equal(pivots, np.array([0, 2]))
    assert np.array_equal(R, np.array([
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ], dtype=np.uint8))


def test_rref_idempotent():
    M = _random_binary((6, 10), seed=7)
    R, _ = gf2_rref(M)
    R2, _ = gf2_rref(R)
    assert np.array_equal(R, R2)


def test_rank_properties_random():
    for seed in range(5):
        A = _random_binary((5, 9), seed=seed)
        r = gf2_rank(A)
        assert r == gf2_rank(A.T)
        assert r <= min(A.shape)
        assert gf2_rank(np.vstack([A, A])) == r


def test_rank_zero_matrix_and_identity():
    assert gf2_rank(np.zeros((3, 4), dtype=np.uint8)) == 0
    assert gf2_rank(np.eye(5, dtype=np.uint8)) == 5


def test_row_basis_rows_independent_and_span():
    H = _random_binary((6, 8), seed=3)
    basis = gf2_row_basis(H)
    assert basis.shape[0] == gf2_rank(H)
    assert gf2_rank(basis) == basis.shape[0]
    for row in H:
        assert in_rowspace(row, basis)


def test_nullspace_basis_is_kernel_basis():
    H = _random_binary((5, 9), seed=11)
    ker = gf2_nullspace_basis(H)
    assert ker.shape == (9 - gf2_rank(H), 9)
    assert not np.any((H.astype(np.int64) @ ker.T.astype(np.int64)) % 2)
    assert gf2_rank(ker) == ker.shape[0]
    # full column rank -> empty kernel with the right shape
    assert gf2_nullspace_basis(np.eye(4, dtype=np.uint8)).shape == (0, 4)


def test_nullspace_non2d_raises():
    with pytest.raises(ValueError, match="must be 2D"):
        gf2_nullspace_basis(np.array([1, 0, 1], dtype=np.uint8))


def test_coset_reps_complete():
    H = _random_binary((4, 7), seed=5)
    reps = gf2_coset_reps_rowspace(H)
    r = gf2_rank(H)
    assert reps.shape == (7 - r, 7)
    stacked = np.vstack([gf2_row_basis(H), reps])
    assert gf2_rank(stacked) == 7


def test_row_span_gray_code_enumeration():
    V = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    span = gf2_row_span(V)
    assert span.shape == (2**3 - 1, 4)
    uniq = {tuple(row) for row in span}
    assert len(uniq) == 7
    assert tuple(np.zeros(4, dtype=np.uint8)) not in uniq
    for row in span:
        assert in_rowspace(row, V)


def test_row_span_empty_input():
    assert gf2_row_span(np.zeros((0, 6), dtype=np.uint8)).shape == (0, 6)


def test_solve_feasible_returns_solution():
    A = _random_binary((6, 4), seed=2)
    x_true = np.array([1, 0, 1, 1], dtype=np.uint8)
    b = (A.astype(np.int64) @ x_true) % 2
    x = gf2_solve(A, b)
    assert x is not None
    assert np.array_equal((A.astype(np.int64) @ x) % 2, b)


def test_solve_infeasible_returns_none():
    A = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    b = np.array([0, 1], dtype=np.uint8)
    assert gf2_solve(A, b) is None


def test_solve_dimension_mismatch_raises():
    A = np.eye(3, dtype=np.uint8)
    with pytest.raises(ValueError, match="dimension mismatch"):
        gf2_solve(A, np.array([1, 0], dtype=np.uint8))


def test_in_rowspace_true_false():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert in_rowspace(np.array([1, 0, 1]), H)
    assert not in_rowspace(np.array([1, 0, 0]), H)


def test_inv_square_roundtrip():
    for seed in range(3):
        A = _random_invertible(6, seed)
        A_inv = _gf2_inv_square(A)
        prod = (A.astype(np.int64) @ A_inv.astype(np.int64)) % 2
        assert np.array_equal(prod, np.eye(6, dtype=np.int64))


def test_inv_square_nonsquare_raises_valueerror():
    with pytest.raises(ValueError, match="must be square"):
        _gf2_inv_square(np.zeros((2, 3), dtype=np.uint8))


def test_inv_square_singular_raises_runtimeerror():
    with pytest.raises(RuntimeError, match="not invertible"):
        _gf2_inv_square(np.zeros((3, 3), dtype=np.uint8))


def test_solve_square_unique_solution():
    A = _random_invertible(5, seed=4)
    B = _random_binary((5, 3), seed=9)
    X = gf2_solve_square(A, B)
    assert X.dtype == np.uint8
    assert np.array_equal((A.astype(np.int64) @ X.astype(np.int64)) % 2, B)
    # the square system has a unique solution, so gf2_solve must agree columnwise
    for j in range(B.shape[1]):
        assert np.array_equal(gf2_solve(A, B[:, j]), X[:, j])


def test_solve_square_error_contract():
    with pytest.raises(ValueError, match="must be square"):
        gf2_solve_square(np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
    A = _random_invertible(3, seed=1)
    with pytest.raises(ValueError, match="must have shape"):
        gf2_solve_square(A, np.array([1, 0, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="must have shape"):
        gf2_solve_square(A, np.zeros((4, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="not full rank"):
        gf2_solve_square(np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8))
