"""Backward-compatibility and golden regression tests for quits.gf2_util.

These tests pin the exact current behavior (outputs, basis forms, report
dicts) of the public gf2_util surface. They must pass unmodified before and
after the module is restructured into a package.
"""

import numpy as np
import pytest


def _steane_h():
    return np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)


def _seeded_8x12():
    return np.random.default_rng(12345).integers(0, 2, size=(8, 12), dtype=np.uint8)


def test_all_legacy_names_importable():
    from quits.gf2_util import (  # noqa: F401
        _as_gf2,
        _gf2_inv_square,
        compute_code_distance,
        compute_lz,
        compute_lz_and_lx,
        gf2_coset_reps_rowspace,
        gf2_nullspace_basis,
        gf2_rank,
        gf2_row_basis,
        gf2_row_span,
        gf2_rref,
        gf2_solve,
        in_rowspace,
        verify_css_logicals,
    )


def test_public_all_contents():
    import quits.gf2_util as gf2_util

    expected = [
        # linalg
        "gf2_coset_reps_rowspace",
        "gf2_nullspace_basis",
        "gf2_rank",
        "gf2_row_basis",
        "gf2_row_span",
        "gf2_rref",
        "gf2_solve",
        "gf2_solve_square",
        "in_rowspace",
        # css
        "check_css",
        "check_weights",
        "compute_code_distance",
        "compute_k",
        "compute_lz",
        "compute_lz_and_lx",
        "qubit_weights",
        "verify_css_logicals",
        # group
        "MultTableGroup",
        "alternating",
        "canonicalize",
        "cyclic",
        "dagger",
        "dihedral",
        "direct_product",
        "element_order",
        "generated_subgroup",
        "is_self_dagger",
        "left_rep",
        "matrix_dagger",
        "right_rep",
        "ring_add",
        "ring_mul",
        "symmetric",
        # lifted_product
        "build_Hx_from_lifted_product",
        "build_Hz_from_lifted_product",
        "get_circulant_mat",
        "lift",
        "lift_enc",
    ]
    assert sorted(gf2_util.__all__) == sorted(expected)


def test_qldpc_util_shim_reexports_same_objects():
    # quits.qldpc_code.qldpc_util is a deprecated re-export location; it must
    # expose the exact same functions as quits.gf2_util
    import quits.gf2_util as gf2_util
    import quits.qldpc_code.qldpc_util as qldpc_util

    assert qldpc_util.get_circulant_mat is gf2_util.get_circulant_mat
    assert qldpc_util.lift is gf2_util.lift
    assert qldpc_util.lift_enc is gf2_util.lift_enc


def test_consumer_modules_import():
    import quits.ldpc_util  # noqa: F401
    import quits.qldpc_code.base  # noqa: F401
    import quits.qldpc_code.bb  # noqa: F401
    import quits.qldpc_code.bpc  # noqa: F401
    import quits.qldpc_code.hgp  # noqa: F401
    import quits.qldpc_code.qlp  # noqa: F401


def test_golden_rref_rank_pivots():
    from quits.gf2_util import gf2_rank, gf2_rref

    M = _seeded_8x12()
    R, pivots = gf2_rref(M)
    assert gf2_rank(M) == 8
    assert isinstance(pivots, np.ndarray)
    assert np.array_equal(pivots, np.array([0, 1, 2, 3, 4, 7, 8, 10]))
    expected_rref = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.uint8)
    assert np.array_equal(R, expected_rref)


def test_golden_nullspace_and_coset_reps():
    from quits.gf2_util import gf2_coset_reps_rowspace, gf2_nullspace_basis

    M = _seeded_8x12()
    expected_nullspace = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    ], dtype=np.uint8)
    assert np.array_equal(gf2_nullspace_basis(M), expected_nullspace)

    expected_reps = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.uint8)
    assert np.array_equal(gf2_coset_reps_rowspace(M), expected_reps)


def test_golden_compute_lz_and_lx_steane():
    from quits.gf2_util import compute_lz_and_lx

    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    expected = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(lz, expected)
    assert np.array_equal(lx, expected)


def test_golden_compute_code_distance():
    from quits.gf2_util import compute_code_distance

    rep5 = (np.eye(4, 5, dtype=np.uint8) + np.eye(4, 5, k=1, dtype=np.uint8))
    assert compute_code_distance(rep5) == 5
    assert compute_code_distance(_steane_h()) == 3
    assert compute_code_distance(np.eye(3, dtype=np.uint8)) == np.inf


def test_golden_verify_report_steane():
    from quits.gf2_util import compute_lz_and_lx, verify_css_logicals

    h = _steane_h()
    lz, lx = compute_lz_and_lx(h, h)
    report = verify_css_logicals(h, h, lz, lx)
    expected = {
        "css_condition": True,
        "lz_commutes_with_X": True,
        "lx_commutes_with_Z": True,
        "rank_hz": 3,
        "rank_hx": 3,
        "rank_lz": 1,
        "rank_lx": 1,
        "k_expected": 1,
        "lz_independent_mod_Z_stabilizers": True,
        "lx_independent_mod_X_stabilizers": True,
        "rank_hz_plus_lz": 4,
        "rank_hx_plus_lx": 4,
        "dim_ker_hz": 4,
        "dim_ker_hx": 4,
        "hx_plus_lx_spans_ker_hz": True,
        "hz_plus_lz_spans_ker_hx": True,
        "same_logicals_ZX_anticommute": True,
        "different_logicals_ZX_commute": True,
        "all_tests_passed": True,
    }
    assert report == expected


def test_golden_compute_lz_and_lx_error_contract():
    from quits.gf2_util import compute_lz_and_lx

    h = _steane_h()
    with pytest.raises(ValueError, match="must be 2D"):
        compute_lz_and_lx(h[0], h)
    with pytest.raises(ValueError, match="same number of columns"):
        compute_lz_and_lx(h, h[:, :5])
    non_css = np.eye(7, dtype=np.uint8)
    with pytest.raises(ValueError, match="Not a CSS pair"):
        compute_lz_and_lx(non_css, non_css)


def test_end_to_end_hgp_code_still_builds():
    from quits.qldpc_code import HgpCode

    d = 3
    H = np.zeros((d, d), dtype=int)
    for i in range(d):
        H[i, i] = 1
        H[i, (i + 1) % d] = 1

    code = HgpCode(H, H)
    assert code.hx.shape == (9, 18)
    assert code.lz.shape == (2, 18)
    report = code.verify_css_logicals()
    assert report["all_tests_passed"]
