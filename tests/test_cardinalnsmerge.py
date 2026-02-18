import numpy as np

from quits.qldpc_code import HgpCode


def test_cardinalnsmerge_builds_ns_group_and_depth():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)

    code.build_circuit(
        strategy="cardinalnsmerge",
        num_rounds=0,
        basis="Z",
        seed=1,
    )

    assert hasattr(code, "colored_edges_NS")
    assert set(code.num_colors.keys()) == {"E", "NS", "W"}

    flattened_ns_edges = [
        edge
        for color in sorted(code.colored_edges_NS.keys())
        for edge in code.colored_edges_NS[color]
    ]
    assert len(flattened_ns_edges) == len(code.edges_N) + len(code.edges_S)
    assert code.depth == code.num_colors["E"] + code.num_colors["NS"] + code.num_colors["W"]
