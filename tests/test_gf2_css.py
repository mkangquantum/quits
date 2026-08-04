"""Unit tests for quits.gf2_util.css (CSS logicals, parameters, distance)."""

import numpy as np
import pytest

from quits.gf2_util import (
    check_css,
    check_weights,
    compute_code_distance,
    compute_k,
    compute_lz,
    compute_lz_and_lx,
    gf2_nullspace_basis,
    gf2_rank,
    in_rowspace,
    qubit_weights,
    verify_css_logicals,
)


def _steane_h():
    return np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)


def test_check_css_true_and_false():
    h = _steane_h()
    assert check_css(h, h)
    assert not check_css(np.array([[1, 0]]), np.array([[1, 0]]))


def test_compute_k_steane_is_one():
    h = _steane_h()
    assert compute_k(h, h) == 1


def test_compute_k_support_normalization():
    # entries >= 2 count as 1 (support semantics), not as their value mod 2
    Hx = np.array([[2, 0]])
    Hz = np.array([[0, 2]])
    assert compute_k(Hx, Hz) == 0


def test_check_weights_and_qubit_weights_known_values():
    h = _steane_h()
    assert check_weights(h, h) == {"Hx_check_weight": 4, "Hz_check_weight": 4}
    qw = qubit_weights(h, h)
    assert qw == {"Hx_qubit_weight": 3, "Hz_qubit_weight": 3, "qubit_weight": 6}


def test_compute_lz_rows_in_kernel_not_in_rowspace():
    h = _steane_h()
    lz = compute_lz(h, h)
    assert lz.shape[0] == 1
    assert not np.any((h.astype(np.int64) @ lz.T.astype(np.int64)) % 2)
    for row in lz:
        assert not in_rowspace(row, h)


def test_compute_lz_and_lx_pairing_identity():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    k = lz.shape[0]
    assert np.array_equal((lz @ lx.T) & 1, np.eye(k, dtype=np.uint8))
    assert not np.any((h @ lz.T) & 1)  # lz in ker(hx)
    assert not np.any((h @ lx.T) & 1)  # lx in ker(hz)


def test_compute_lz_and_lx_k_zero_returns_empty():
    # [[2,0],[0,2]] surface-less pair: hz = hx = I is not CSS unless n allows;
    # use hz = [1 0], hx = [0 1] -> n=2, ranks 1+1 -> k=0, and hx @ hz.T = 0.
    hz = np.array([[1, 0]], dtype=np.uint8)
    hx = np.array([[0, 1]], dtype=np.uint8)
    lz, lx = compute_lz_and_lx(hz, hx)
    assert lz.shape == (0, 2)
    assert lx.shape == (0, 2)


def test_compute_lz_and_lx_error_contract():
    h = _steane_h()
    with pytest.raises(ValueError, match="must be 2D"):
        compute_lz_and_lx(h[0], h)
    with pytest.raises(ValueError, match="same number of columns"):
        compute_lz_and_lx(h, h[:, :5])
    with pytest.raises(ValueError, match="Not a CSS pair"):
        compute_lz_and_lx(np.eye(7, dtype=np.uint8), np.eye(7, dtype=np.uint8))


def test_verify_css_logicals_all_pass_on_computed():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    report = verify_css_logicals(h, h, lz, lx)
    assert report["all_tests_passed"] is True


def test_verify_css_logicals_detects_corruption():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    bad_lz = lz.copy()
    bad_lz[0, 6] ^= 1  # now anticommutes with X stabilizers
    report = verify_css_logicals(h, h, bad_lz, lx)
    assert report["lz_commutes_with_X"] is False
    assert report["all_tests_passed"] is False


def test_verify_css_logicals_report_keys_stable():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    report = verify_css_logicals(h, h, lz, lx)
    expected_keys = {
        "css_condition",
        "lz_commutes_with_X",
        "lx_commutes_with_Z",
        "rank_hz",
        "rank_hx",
        "rank_lz",
        "rank_lx",
        "k_expected",
        "lz_independent_mod_Z_stabilizers",
        "lx_independent_mod_X_stabilizers",
        "rank_hz_plus_lz",
        "rank_hx_plus_lx",
        "dim_ker_hz",
        "dim_ker_hx",
        "hx_plus_lx_spans_ker_hz",
        "hz_plus_lz_spans_ker_hx",
        "same_logicals_ZX_anticommute",
        "different_logicals_ZX_commute",
        "all_tests_passed",
    }
    assert set(report.keys()) == expected_keys


def test_verify_css_logicals_column_mismatch_raises():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    with pytest.raises(ValueError, match="same number of columns"):
        verify_css_logicals(h, h[:, :5], lz, lx)


def test_compute_code_distance_repetition_and_hamming():
    rep5 = (np.eye(4, 5, dtype=np.uint8) + np.eye(4, 5, k=1, dtype=np.uint8))
    assert compute_code_distance(rep5) == 5
    assert compute_code_distance(_steane_h()) == 3
    assert compute_code_distance(np.eye(3, dtype=np.uint8)) == np.inf


def test_k_consistency_between_helpers():
    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    assert lz.shape[0] == compute_k(h, h)
    assert compute_k(h, h) == h.shape[1] - 2 * gf2_rank(h)
    assert gf2_nullspace_basis(h).shape[0] == h.shape[1] - gf2_rank(h)
