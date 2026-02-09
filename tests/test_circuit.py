import numpy as np
import stim

from quits.circuit import check_overlapping_CX
from quits.noise import ErrorModel
from quits.qldpc_code.circuit_construction.cardinal import CardinalBuilder
from quits.qldpc_code import HgpCode


def test_check_overlapping_cx_hgp_prints_when_verbose():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    code.build_circuit(strategy="cardinal", seed=22)
    em = ErrorModel(1e-3, 1e-3, 1e-3, 1e-3)

    circuit = stim.Circuit(
        CardinalBuilder(code).get_cardinal_circuit(error_model=em, num_rounds=1, basis="Z")
    )

    overlaps = check_overlapping_CX(circuit, verbose=True)
    print({"overlaps": [(i, d.tolist()) for i, d in overlaps]})

    assert overlaps == []
