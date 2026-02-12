import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder

from quits.noise import ErrorModel
from quits.decoder.sliding_window import sliding_window_phenom_mem
from quits.qldpc_code import BbCode, BpcCode, HgpCode, LscCode, QlpCode
from quits.simulation import get_stim_mem_result


def _simulate_mem_cardinal_circuit(code, p, num_rounds, num_trials, basis="Z", seed=1):
    em = ErrorModel(p, p, p, p)
    circuit = code.build_circuit(
        strategy="cardinal",
        error_model=em,
        num_rounds=num_rounds,
        basis=basis,
        seed=seed,
    )
    detection_events, observable_flips = get_stim_mem_result(
        circuit,
        num_trials,
        seed=1,
    )
    return circuit, detection_events, observable_flips


def _simulate_mem_zxcoloration_circuit(code, p, num_rounds, num_trials, basis="Z"):
    em = ErrorModel(p, p, p, p)
    circuit = code.build_circuit(
        strategy="zxcoloration",
        error_model=em,
        num_rounds=num_rounds,
        basis=basis,
    )
    detection_events, observable_flips = get_stim_mem_result(
        circuit,
        num_trials,
        seed=1,
    )
    return circuit, detection_events, observable_flips


def _simulate_mem_custom_circuit(code, p, num_rounds, num_trials, basis="Z"):
    em = ErrorModel(p, p, p, p)
    circuit = code.build_circuit(
        strategy="custom",
        error_model=em,
        num_rounds=num_rounds,
        basis=basis,
    )
    detection_events, observable_flips = get_stim_mem_result(
        circuit,
        num_trials,
        seed=1,
    )
    return circuit, detection_events, observable_flips


def _bp_osd_params(max_iter, osd_order):
    return {
        "bp_method": "product_sum",
        "max_iter": max_iter,
        "schedule": "serial",
        "osd_method": "osd_cs",
        "osd_order": osd_order,
    }


def _run_sliding_window_phenom(
    code,
    code_name,
    p,
    num_rounds,
    num_trials,
    W,
    F,
    max_iter,
    osd_order,
    seed=1,
    strategy="cardinal",
):
    if strategy == "cardinal":
        code.build_circuit(strategy="cardinal", seed=seed)
    elif strategy == "zxcoloration":
        code.build_circuit(strategy="zxcoloration")
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    report = code.verify_css_logicals()
    print(f"{code_name} verify_css_logicals", report)
    assert report["all_tests_passed"]

    depth = code.depth
    eff_error_rate_per_fault = p * (depth + 3)

    if strategy == "cardinal":
        _, detection_events, observable_flips = _simulate_mem_cardinal_circuit(
            code,
            p,
            num_rounds,
            num_trials,
            seed=seed,
        )
    else:
        _, detection_events, observable_flips = _simulate_mem_zxcoloration_circuit(
            code,
            p,
            num_rounds,
            num_trials,
        )

    dict1 = _bp_osd_params(max_iter, osd_order)
    dict2 = _bp_osd_params(max_iter, osd_order)
    dict1["error_rate"] = float(eff_error_rate_per_fault)
    dict2["error_rate"] = float(eff_error_rate_per_fault)

    logical_pred = sliding_window_phenom_mem(
        detection_events,
        code.hz,
        code.lz,
        W,
        F,
        BpOsdDecoder,
        BpOsdDecoder,
        dict1,
        dict2,
        "decode",
        "decode",
        tqdm_on=False,
    )

    pL = np.mean((observable_flips - logical_pred).any(axis=1))
    lfr = 1 - (1 - pL) ** (1 / num_rounds)
    return depth, eff_error_rate_per_fault, pL, lfr


def _print_results(code_name, params, depth, eff_error_rate_per_fault, pL, lfr):
    print(
        f"{code_name} sliding_window_phenom_mem",
        {
            **params,
            "depth": depth,
            "eff_error_rate_per_fault": float(eff_error_rate_per_fault),
            "pL": float(pL),
            "lfr": float(lfr),
        },
    )


def test_hgp_code_circuit_low_lfr():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)

    params = {
        "p": 5e-4,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_sliding_window_phenom(code, "HGP", **params, seed=1)
    _print_results("HGP", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.25
    assert lfr <= 0.08


def test_hgpcode_zxcoloration_circuit_low_lfr():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)

    params = {
        "p": 5e-4,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_sliding_window_phenom(
        code,
        "HGP-ZX",
        **params,
        strategy="zxcoloration",
    )
    _print_results("HGP-ZX", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.25
    assert lfr <= 0.08


def test_hgp_toric_d3_non_full_rank_canonical_logicals_shape():
    d = 3
    H = np.zeros((d, d), dtype=int)
    for i in range(d):
        H[i, i] = 1
        H[i, (i + 1) % d] = 1

    code = HgpCode(H, H)

    assert code.hx.shape == (9, 18)
    assert code.hz.shape == (9, 18)
    assert code.lx.shape == (2, 18)
    assert code.lz.shape == (2, 18)

    report = code.verify_css_logicals()
    assert report["all_tests_passed"]


def test_qlp_code_circuit_low_lfr():
    lift_size = 16
    b = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 4, 7, 11],
            [0, 3, 10, 14, 15],
        ]
    )
    code = QlpCode(b, b, lift_size)

    params = {
        "p": 5e-4,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_sliding_window_phenom(code, "QLP", **params)
    _print_results("QLP", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.3
    assert lfr <= 0.1


def test_bpc_code_circuit_low_lfr():
    lift_size, factor = 15, 3
    p1 = [0, 1, 5]
    p2 = [0, 8, 13]
    code = BpcCode(p1, p2, lift_size, factor)

    params = {
        "p": 1e-3,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_sliding_window_phenom(code, "BPC", **params)
    _print_results("BPC", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.3
    assert lfr <= 0.1


def test_lsc_code_circuit_low_lfr():
    lift_size = 5
    length = 3  # length = l + 1
    code = LscCode(lift_size, length)

    expected_b = np.array(
        [
            [[0], [0, 1], []],
            [[], [0], [0, 1]],
        ],
        dtype=object,
    )
    assert (code.b == expected_b).all()

    params = {
        "p": 1e-3,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_sliding_window_phenom(code, "LSC", **params, seed=1)
    _print_results("LSC", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.3
    assert lfr <= 0.1


def test_bb_code_circuit_low_lfr():
    code = BbCode(
        l=15,
        m=3,
        A_x_pows=[9],
        A_y_pows=[1, 2],
        B_x_pows=[2, 7],
        B_y_pows=[0],
    )

    report = code.verify_css_logicals()
    print("BB verify_css_logicals", report)
    assert report["all_tests_passed"]

    params = {
        "p": 1e-3,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    _, detection_events, observable_flips = _simulate_mem_custom_circuit(
        code,
        p=params["p"],
        num_rounds=params["num_rounds"],
        num_trials=params["num_trials"],
    )

    depth = 7
    eff_error_rate_per_fault = params["p"] * (depth + 3)
    dict1 = _bp_osd_params(params["max_iter"], params["osd_order"])
    dict2 = _bp_osd_params(params["max_iter"], params["osd_order"])
    dict1["error_rate"] = float(eff_error_rate_per_fault)
    dict2["error_rate"] = float(eff_error_rate_per_fault)

    logical_pred = sliding_window_phenom_mem(
        detection_events,
        code.hz,
        code.lz,
        params["W"],
        params["F"],
        BpOsdDecoder,
        BpOsdDecoder,
        dict1,
        dict2,
        "decode",
        "decode",
        tqdm_on=False,
    )

    pL = np.mean((observable_flips - logical_pred).any(axis=1))
    lfr = 1 - (1 - pL) ** (1 / params["num_rounds"])
    print(
        "BB sliding_window_phenom_mem",
        {
            **params,
            "depth": depth,
            "eff_error_rate_per_fault": float(eff_error_rate_per_fault),
            "pL": float(pL),
            "lfr": float(lfr),
        },
    )
    print()

    assert pL <= 0.3
    assert lfr <= 0.1
