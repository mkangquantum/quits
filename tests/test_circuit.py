import numpy as np
import pytest
import stim
from quits.circuit import check_overlapping_CX
from quits.noise import ErrorModel
from quits.qldpc_code import BbCode, HgpCode, LcsCode, QlpCode


@pytest.mark.parametrize("strategy", ["cardinal", "cardinalnsmerge"])
def test_check_overlapping_cx_hgp_prints_when_verbose(strategy):
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    em = ErrorModel(5e-4, 5e-4, 5e-4, 5e-4)

    circuit = code.build_circuit(
        strategy=strategy,
        error_model=em,
        num_rounds=1,
        basis="Z",
        seed=1,
    )

    overlaps = check_overlapping_CX(circuit, verbose=True)
    print({"overlaps": [(i, d.tolist()) for i, d in overlaps]})

    assert overlaps == []
