import numpy as np
import stim

from quits.circuit import get_qldpc_mem_circuit
from quits.decoder import sliding_window_bplsd_phenom_mem, sliding_window_bposd_phenom_mem
from quits.qldpc_code import HgpCode
from quits.simulation import get_stim_mem_result


def _build_hgp_code(seed=22):
    h = np.loadtxt(
        "parity_check_matrices/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    code.build_circuit(strategy="cardinal", seed=seed)
    return code


def _simulate_mem_circuit(code, p, num_rounds, num_trials, basis="Z"):
    circuit = stim.Circuit(
        get_qldpc_mem_circuit(
            code,
            p,
            p,
            p,
            p,
            num_rounds,
            basis=basis,
        )
    )
    detection_events, observable_flips = get_stim_mem_result(
        circuit,
        num_trials,
        seed=1,
    )
    return circuit, detection_events, observable_flips


def _run_phenom_decoder(
    decoder_fn,
    decoder_params,
    code,
    p,
    num_rounds,
    num_trials,
    W,
    F,
):
    depth = sum(code.num_colors.values())
    eff_error_rate_per_fault = p * (depth + 3)

    _, detection_events, observable_flips = _simulate_mem_circuit(
        code,
        p,
        num_rounds,
        num_trials,
    )

    logical_pred = decoder_fn(
        detection_events,
        code.hz,
        code.lz,
        W,
        F,
        eff_error_rate_per_fault=eff_error_rate_per_fault,
        tqdm_on=False,
        **decoder_params,
    )

    pL = np.mean((observable_flips - logical_pred).any(axis=1))
    lfr = 1 - (1 - pL) ** (1 / num_rounds)
    return depth, eff_error_rate_per_fault, pL, lfr


def _print_results(name, params, depth, eff_error_rate_per_fault, pL, lfr):
    print(
        f"{name} sliding_window_phenom_mem",
        {
            **params,
            "depth": depth,
            "eff_error_rate_per_fault": float(eff_error_rate_per_fault),
            "pL": float(pL),
            "lfr": float(lfr),
        },
    )


def test_bposd_decoder_phenom_low_lfr():
    code = _build_hgp_code(seed=22)
    report = code.verify_css_logicals()
    assert report["ok"]

    params = {
        "p": 1e-3,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "osd_order": 1,
    }

    decoder_params = {
        "max_iter": params["max_iter"],
        "osd_order": params["osd_order"],
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_phenom_decoder(
        sliding_window_bposd_phenom_mem,
        decoder_params,
        code,
        params["p"],
        params["num_rounds"],
        params["num_trials"],
        params["W"],
        params["F"],
    )
    _print_results("BPOSD", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.25
    assert lfr <= 0.08


def test_bplsd_decoder_phenom_low_lfr():
    code = _build_hgp_code(seed=22)
    report = code.verify_css_logicals()
    assert report["ok"]

    params = {
        "p": 1e-3,
        "num_rounds": 15,
        "num_trials": 50,
        "W": 5,
        "F": 3,
        "max_iter": 10,
        "lsd_order": 1,
    }

    decoder_params = {
        "max_iter": params["max_iter"],
        "lsd_order": params["lsd_order"],
    }

    depth, eff_error_rate_per_fault, pL, lfr = _run_phenom_decoder(
        sliding_window_bplsd_phenom_mem,
        decoder_params,
        code,
        params["p"],
        params["num_rounds"],
        params["num_trials"],
        params["W"],
        params["F"],
    )
    _print_results("BPLSD", params, depth, eff_error_rate_per_fault, pL, lfr)
    print()

    assert pL <= 0.3
    assert lfr <= 0.1
