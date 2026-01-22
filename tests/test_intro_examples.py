import numpy as np

from quits.qldpc_code import BpcCode, HgpCode, QlpCode


def test_hgp_code_intro_example():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    code.build_graph(seed=22)

    report = code.verify_css_logicals()
    assert report["ok"]
    assert report["k_expected"] == code.lz.shape[0] == code.lx.shape[0]
    assert sum(code.num_colors.values()) == 8


def test_qlp_code_intro_example():
    lift_size = 16
    b = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 4, 7, 11],
            [0, 3, 10, 14, 15],
        ]
    )
    code = QlpCode(b, b, lift_size)
    code.build_graph(seed=1)

    report = code.verify_css_logicals()
    assert report["ok"]
    assert report["k_expected"] == code.lz.shape[0] == code.lx.shape[0]


def test_bpc_code_intro_example():
    lift_size, factor = 15, 3
    p1 = [0, 1, 5]
    p2 = [0, 8, 13]
    code = BpcCode(p1, p2, lift_size, factor)
    code.build_graph(seed=1)

    report = code.verify_css_logicals()
    assert report["ok"]
    assert report["k_expected"] == code.lz.shape[0] == code.lx.shape[0]
