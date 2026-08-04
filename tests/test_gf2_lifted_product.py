"""Unit tests for quits.gf2_util.lifted_product (circulant lifts, LP codes).

Golden values pin the behavior of the reference implementation of
arXiv:2607.28795 bit-for-bit.
"""

import numpy as np
import pytest

from quits.gf2_util import (
    build_Hx_from_lifted_product,
    build_Hz_from_lifted_product,
    check_css,
    check_weights,
    compute_k,
    cyclic,
    dihedral,
    get_circulant_mat,
    left_rep,
    lift,
    lift_enc,
    qubit_weights,
    symmetric,
)
from quits.gf2_util.lifted_product import _check_rectangular

# Fixed S3 lifted-product example (valid LP pair, NOT a valid mitten pair).
S3_A = [[(0, 1, 3), (0, 2, 4)]]
S3_B = [[(0, 1, 2), (0, 3, 5)]]


def _random_ring_matrix(gd, rng, shape, weight=3):
    return [[tuple(sorted(int(g) for g in rng.choice(gd.n, size=weight, replace=False)))
             for _ in range(shape[1])] for _ in range(shape[0])]


def test_build_hx_hz_shapes_s3_example():
    G = symmetric(3)
    Hx = build_Hx_from_lifted_product(S3_A, S3_B, G)
    Hz = build_Hz_from_lifted_product(S3_A, S3_B, G)
    assert Hx.shape == (12, 30)
    assert Hz.shape == (12, 30)


def test_lifted_product_css_condition_multiple_groups():
    # Hx @ Hz.T = 0 is an algebraic identity of the construction: it holds
    # for arbitrary base matrices, not just valid mitten pairs.
    for G, seed in ((cyclic(5), 0), (symmetric(3), 1), (dihedral(4), 2)):
        rng = np.random.default_rng(seed)
        A = _random_ring_matrix(G, rng, (1, 2))
        B = _random_ring_matrix(G, rng, (1, 2))
        assert check_css(build_Hx_from_lifted_product(A, B, G),
                         build_Hz_from_lifted_product(A, B, G))


def test_s3_example_known_answer():
    G = symmetric(3)
    Hx = build_Hx_from_lifted_product(S3_A, S3_B, G)
    Hz = build_Hz_from_lifted_product(S3_A, S3_B, G)
    assert check_css(Hx, Hz)
    assert compute_k(Hx, Hz) == 8
    assert check_weights(Hx, Hz) == {"Hx_check_weight": 9, "Hz_check_weight": 9}
    assert qubit_weights(Hx, Hz) == {
        "Hx_qubit_weight": 6,
        "Hz_qubit_weight": 6,
        "qubit_weight": 12,
    }


def test_check_rectangular_raises_on_ragged_input():
    with pytest.raises(ValueError, match="not rectangular"):
        _check_rectangular([[(0,), (1,)], [(0,)]], "A")


def test_get_circulant_mat_known_answers():
    # power 1: shift by one, column h has its 1 in row h+1 (mod size)
    S = get_circulant_mat(3, 1)
    assert S.dtype == np.dtype(int)
    assert np.array_equal(S, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    # power 0 is the identity; negative powers wrap
    assert np.array_equal(get_circulant_mat(3, 0), np.eye(3, dtype=int))
    assert np.array_equal(get_circulant_mat(3, -1), get_circulant_mat(3, 2))
    # matches the left regular representation of the cyclic group
    C4 = cyclic(4)
    for power in range(4):
        assert np.array_equal(get_circulant_mat(4, power), left_rep((power,), C4))


def test_lift_known_answer():
    h_base = np.array([[0, 1]], dtype=int)
    placeholder = np.array([[1, 1]], dtype=int)
    h = lift(2, h_base, placeholder)
    assert np.array_equal(h, np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]))
    # zeroed placeholder entries produce zero blocks even for nonzero powers
    h = lift(2, np.array([[1]], dtype=int), np.array([[0]], dtype=int))
    assert np.array_equal(h, np.zeros((2, 2), dtype=int))


def test_lift_enc_polynomial_known_answer():
    # encoded entry 6 in base 4 has digits [2, 1] -> x^2 + x^1
    h = lift_enc(4, np.array([[6]], dtype=int), np.array([[1]], dtype=int))
    expected = get_circulant_mat(4, 2) + get_circulant_mat(4, 1)
    assert np.array_equal(h, expected)
    # encoded entry 0 with a placeholder 1 means the constant term x^0
    h = lift_enc(3, np.array([[0]], dtype=int), np.array([[1]], dtype=int))
    assert np.array_equal(h, np.eye(3, dtype=int))
