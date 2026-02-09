import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder

from quits.noise import ErrorModel
from quits.decoder.sliding_window import sliding_window_circuit_mem, sliding_window_phenom_mem
from quits.qldpc_code import HgpCode
from quits.simulation import get_stim_mem_result


def _build_hgp_code():
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    code.build_circuit(strategy="cardinal", seed=22)
    return code


def _simulate_mem_circuit(code, p, num_rounds, num_trials, basis="Z", seed=22):
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


def _bp_osd_params(max_iter, osd_order):
    return {
        "bp_method": "product_sum",
        "max_iter": max_iter,
        "schedule": "serial",
        "osd_method": "osd_cs",
        "osd_order": osd_order,
    }


def test_sliding_window_circuit_mem_low_logical_error():
    code = _build_hgp_code()
    p = 1e-3
    num_rounds = 15
    num_trials = 50
    W, F = 5, 3
    max_iter, osd_order = 10, 1

    circuit, detection_events, observable_flips = _simulate_mem_circuit(
        code,
        p,
        num_rounds,
        num_trials,
        seed=22,
    )

    dict1 = _bp_osd_params(max_iter, osd_order)
    dict2 = _bp_osd_params(max_iter, osd_order)

    logical_pred = sliding_window_circuit_mem(
        detection_events,
        circuit,
        code.hz,
        code.lz,
        W,
        F,
        BpOsdDecoder,
        BpOsdDecoder,
        dict1,
        dict2,
        "channel_probs",
        "channel_probs",
        "decode",
        "decode",
        tqdm_on=False,
    )

    pL = np.mean((observable_flips - logical_pred).any(axis=1))
    lfr = 1 - (1 - pL) ** (1 / num_rounds)

    print(
        "sliding_window_circuit_mem",
        {
            "p": p,
            "num_rounds": num_rounds,
            "num_trials": num_trials,
            "W": W,
            "F": F,
            "max_iter": max_iter,
            "osd_order": osd_order,
            "pL": float(pL),
            "lfr": float(lfr),
        },
    )
    print()

    assert pL <= 0.2
    assert lfr <= 0.05


def test_sliding_window_phenom_mem_low_logical_error():
    code = _build_hgp_code()
    p = 1e-3
    num_rounds = 15
    num_trials = 50
    W, F = 5, 3
    max_iter, osd_order = 10, 1

    circuit, detection_events, observable_flips = _simulate_mem_circuit(
        code,
        p,
        num_rounds,
        num_trials,
        seed=22,
    )

    depth = sum(code.num_colors.values())
    eff_error_rate_per_fault = p * (depth + 3)

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

    print(
        "sliding_window_phenom_mem",
        {
            "p": p,
            "num_rounds": num_rounds,
            "num_trials": num_trials,
            "W": W,
            "F": F,
            "max_iter": max_iter,
            "osd_order": osd_order,
            "depth": depth,
            "eff_error_rate_per_fault": float(eff_error_rate_per_fault),
            "pL": float(pL),
            "lfr": float(lfr),
        },
    )

    assert pL <= 0.25
    assert lfr <= 0.08
