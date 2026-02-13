import warnings

import numpy as np
import pytest

from quits.qldpc_code import BbCode, BpcCode, QldpcCode, QlpCode
from quits.qldpc_code.qldpc_util import get_circulant_mat, lift, lift_enc


def test_get_circulant_mat_matches_legacy_wrapper():
    code = QldpcCode()
    with pytest.deprecated_call(match="QldpcCode.get_circulant_mat is deprecated"):
        legacy = code.get_circulant_mat(5, -1)
    direct = get_circulant_mat(5, -1)
    assert np.array_equal(direct, legacy)


def test_lift_matches_legacy_wrapper():
    code = QldpcCode()
    h_base = np.array([[0, 1], [2, 3]], dtype=int)
    h_base_placeholder = np.array([[1, 0], [1, 1]], dtype=int)
    with pytest.deprecated_call(match="QldpcCode.lift is deprecated"):
        legacy = code.lift(5, h_base, h_base_placeholder)
    direct = lift(5, h_base, h_base_placeholder)
    assert np.array_equal(direct, legacy)


def test_lift_enc_matches_legacy_wrapper():
    code = QldpcCode()
    h_base_enc = np.array([[0, 6], [2, 0]], dtype=int)
    h_base_placeholder = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.deprecated_call(match="QldpcCode.lift_enc is deprecated"):
        legacy = code.lift_enc(4, h_base_enc, h_base_placeholder)
    direct = lift_enc(4, h_base_enc, h_base_placeholder)
    assert np.array_equal(direct, legacy)


def test_legacy_wrappers_emit_deprecation_warnings():
    code = QldpcCode()
    with pytest.deprecated_call(match="QldpcCode.get_circulant_mat is deprecated"):
        code.get_circulant_mat(4, 1)
    with pytest.deprecated_call(match="QldpcCode.lift is deprecated"):
        code.lift(3, np.array([[0]], dtype=int), np.array([[1]], dtype=int))
    with pytest.deprecated_call(match="QldpcCode.lift_enc is deprecated"):
        code.lift_enc(3, np.array([[0]], dtype=int), np.array([[1]], dtype=int))


def test_internal_construction_paths_do_not_use_deprecated_wrappers():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", DeprecationWarning)

        bpc = BpcCode(p1=[0, 1, 2], p2=[0, 4, 5], lift_size=6, factor=3)
        bpc.build_circuit(strategy="cardinal", num_rounds=0, basis="Z", seed=1)

        qlp_base = np.array([[0, 0], [0, 1]], dtype=int)
        qlp = QlpCode(qlp_base, qlp_base, lift_size=4)
        qlp.build_circuit(strategy="cardinal", num_rounds=0, basis="Z", seed=1)

        bb = BbCode(
            l=15,
            m=3,
            A_x_pows=[9],
            A_y_pows=[1, 2],
            B_x_pows=[2, 7],
            B_y_pows=[0],
        )
        bb.build_circuit(strategy="custom", num_rounds=0, basis="Z")

    deprecated_msgs = [
        str(w.message)
        for w in rec
        if (
            "QldpcCode.get_circulant_mat is deprecated" in str(w.message)
            or "QldpcCode.lift is deprecated" in str(w.message)
            or "QldpcCode.lift_enc is deprecated" in str(w.message)
        )
    ]
    assert deprecated_msgs == []
