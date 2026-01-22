"""
@author: Yingjia Lin
"""

import warnings

import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm

from .base import spacetime


def sliding_window_phenom_mem(zcheck_samples, hz, lz, W, F, decoder1, decoder2, dict1: dict, dict2: dict, function_name1: str, function_name2: str, tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024)
    This version allows to replace the decoder used in the decoding scheme.
    The space-time code has not yet been implemented

    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory.

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param decoder1: A python class for an inner decoder for each window before the last window. This class can be initialized by some parameters and has a decode function that takes a syndrome and returns a correction.
    :param decoder2: A python class for an inner decoder for the last window including the ideal round of measurement
    :param dict1: parameters of decoder1, except for the parity check matrix
    :param dict2: parameters of decoder2, except for the parity check matrix
    :param function_name1: the decoding function name of decoder1 that is called to decode a syndrome after the initialization of the decoder1
    :param function_name2: the decoding function of decoder1 that is called to decode a syndrome after the initialization of the decoder2
    :param eff_error_rate_per_fault: Estimate of error rate in the context of code-capacity/phenomenological level simulations.
                For circuit level, we suggest p * (num_layers + 3), where p is the depolarizing error rate
                and num_layers is the circuit depth for each stabilizer measurement round
                e.g. for hgp codes, num_layers == code.count_color('east') + code.count_color('north') + code.count_color('south') + code.count_color('west')

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''
    if F == 0:
        raise ValueError("Input parameter F cannot be zero.")
    num_trials = zcheck_samples.shape[0]
    num_rounds = zcheck_samples.shape[1] // hz.shape[0] - 2

    # update the total number of windows for decoding, the size of the last window
    if 2 + num_rounds - W >= 0:
        num_cor_rounds = (2 + num_rounds - W) // F  # num_cor_rounds=num of windows before the last window
        if (2 + num_rounds - W) % F != 0:  # we can slide one more window if the remaining rounds>W
            num_cor_rounds += 1
    else:
        num_cor_rounds = 0
        warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
    W_last = num_rounds + 2 - F * num_cor_rounds

    # update the window matrix and the decoder
    B = np.eye(W, dtype=int)
    for i in range(1, W):
        B[i, i - 1] = 1
    hz_phenom = np.column_stack((np.kron(np.eye(W, dtype=int), hz), np.kron(B, np.eye(hz.shape[0], dtype=int))))

    decoder_each_window = decoder1(csc_matrix(hz_phenom), **dict1)

    B_last = np.eye(W_last, dtype=int)
    for i in range(1, W_last):
        B_last[i, i - 1] = 1
    B_last = B_last[:, :W_last - 1]  # The last round in this window is ideal

    hz_last = np.column_stack((np.kron(np.eye(W_last, dtype=int), hz), np.kron(B_last, np.eye(hz.shape[0], dtype=int))))
    decoder_last_window = decoder2(csc_matrix(hz_last), **dict2)

    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)
    logical_z_pred = np.zeros((num_trials, lz.shape[0]), dtype=int)

    for i in iterator:  # each sample decoding
        accumulated_correction = np.zeros(hz.shape[1], dtype=int)
        syn_update = np.zeros(hz.shape[0], dtype=int)

        for k in range(num_cor_rounds):
            diff_syndrome = (zcheck_samples[i, F * k * hz.shape[0]:(F * k + W) * hz.shape[0]].copy()) % 2
            diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2

            decoded_errors = getattr(decoder_each_window, function_name1)(diff_syndrome)
            correction = np.sum(decoded_errors[:F * hz.shape[1]].reshape(F, hz.shape[1]).copy(), axis=0) % 2

            syn_update = decoded_errors[W * hz.shape[1] + (F - 1) * hz.shape[0]:W * hz.shape[1] + F * hz.shape[0]].copy()
            accumulated_correction = (accumulated_correction + correction) % 2

        # In the last round we just correct the whole window
        diff_syndrome = (zcheck_samples[i, (F * num_cor_rounds) * hz.shape[0]:].copy()) % 2
        diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2

        decoded_errors = getattr(decoder_last_window, function_name2)(diff_syndrome)
        correction = np.sum(decoded_errors[:W_last * hz.shape[1]].reshape(W_last, hz.shape[1]), axis=0) % 2

        accumulated_correction = (accumulated_correction + correction) % 2
        logical_z_pred[i, :] = (lz @ accumulated_correction) % 2

    return logical_z_pred


def sliding_window_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, decoder1, decoder2, dict1: dict, dict2: dict,
                               error_rate_name1: str, error_rate_name2: str,
                               function_name1: str, function_name2: str, tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024) implemented with spacetime
    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory.

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param circuit: simulated stim.Circuit
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param decoder1: A python class for an inner decoder for each window before the last window. This class can be initialized by some parameters and has a decode function that takes a syndrome and returns a correction.
    :param decoder2: A python class for an inner decoder for the last window including the ideal round of measurement
    :param dict1: parameters of decoder1, except for the parity check matrix
    :param dict2: parameters of decoder2, except for the parity check matrix
    :param error_rate_name1: the parameter name of the list of error rates for decoder1 for initializing decoder1
    :param error_rate_name2: the parameter name of the list of error rates for decoder2 for initializing decoder2
    :param function_name1: the decoding function name of decoder1 that is called to decode a syndrome after the initialization of the decoder1
    :param function_name2: the decoding function name of decoder2 that is called to decode a syndrome after the initialization of the decoder2
    :param tqdm_on: True or False: Evaluating the iteration runtime

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''

    num_trials = zcheck_samples.shape[0]
    num_rounds = zcheck_samples.shape[1] // hz.shape[0] - 2

    # update the total number of windows for decoding, the size of the last window
    if 2 + num_rounds - W >= 0:
        num_cor_rounds = (2 + num_rounds - W) // F  # num_cor_rounds=num of windows before the last window
        if (2 + num_rounds - W) % F != 0:  # we can slide one more window if the remaining rounds>W
            num_cor_rounds += 1
    else:
        num_cor_rounds = 0
        warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
    W_last = num_rounds + 2 - F * num_cor_rounds
    # update the window matrix and the decoder
    # spacetime detector error matrix
    window_check_set, window_observable_set, window_priors_set, window_update = spacetime(circuit, hz, W, F, num_cor_rounds)
    # decoder for each window
    decoder = []
    for i in range(len(window_check_set) - 1):
        dict1[error_rate_name1] = window_priors_set[i]
        decoder_each_window = decoder1(window_check_set[i], **dict1)
        decoder.append(decoder_each_window)
    dict2[error_rate_name2] = window_priors_set[len(window_check_set) - 1]
    decoder_each_window = decoder2(window_check_set[len(window_check_set) - 1], **dict2)
    decoder.append(decoder_each_window)

    # start decoding
    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)
    logical_z_pred = np.zeros((num_trials, lz.shape[0]), dtype=int)

    for i in iterator:  # each sample decoding
        accumulated_correction = np.zeros(window_observable_set[0].shape[0], dtype=int)
        syn_update = np.zeros(hz.shape[0], dtype=int)

        for k in range(num_cor_rounds):
            # syndrome of the window
            diff_syndrome = (zcheck_samples[i, F * k * hz.shape[0]:(F * k + W) * hz.shape[0]].copy()) % 2
            diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2  # update the syndrome based on the previous window decoding

            decoded_errors = getattr(decoder[k], function_name1)(diff_syndrome)
            correction = window_observable_set[k] @ decoded_errors[:window_observable_set[k].shape[1]] % 2  # interpret the correction operation as final observable flips

            syn_update = window_update[k] @ decoded_errors[:window_observable_set[k].shape[1]] % 2
            accumulated_correction = (accumulated_correction + correction) % 2

        # In the last round we just correct the whole window
        # syndrome of last round
        diff_syndrome = (zcheck_samples[i, (F * num_cor_rounds) * hz.shape[0]:].copy()) % 2
        diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2
        # Observable flips based on correction
        decoded_errors = getattr(decoder[num_cor_rounds], function_name2)(diff_syndrome)
        correction = window_observable_set[num_cor_rounds] @ decoded_errors % 2
        accumulated_correction = (accumulated_correction + correction) % 2
        # Predicted observable flips
        logical_z_pred[i, :] = accumulated_correction

    return logical_z_pred


__all__ = ["sliding_window_phenom_mem", "sliding_window_circuit_mem"]
