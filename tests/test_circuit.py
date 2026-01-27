import numpy as np
import stim

from quits.circuit import check_overlapping_CX, get_qldpc_mem_circuit
from quits.qldpc_code import HgpCode


def test_check_overlapping_cx_hgp_prints_when_verbose():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    code.build_circuit(strategy="cardinal", seed=22)

    circuit = stim.Circuit(
        get_qldpc_mem_circuit(code, 1e-3, 1e-3, 1e-3, 1e-3, 1, basis="Z")
    )

    overlaps = check_overlapping_CX(circuit, verbose=True)
    print({"overlaps": [(i, d.tolist()) for i, d in overlaps]})

    assert overlaps == []
