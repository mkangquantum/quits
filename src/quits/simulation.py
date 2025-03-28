"""
@author: Mingyu Kang
"""

import numpy as np
from tqdm import tqdm

def get_stim_mem_result(circuit, num_trials, seed=-1):
    '''
    Simulates the Stim circuit and obtains the detector and logical observable flip results for logical memory. 

    :param circuit: Stim circuit
    :param num_trials: Number of trials to sample from
    :param seed: Random seed

    :return detections: 2-dim numpy array of shape (num_trials, # detectors). Each entry is a bit (0 or 1) that represents the detector result. 
                        If get_all_detections, # detectors = num_(basis)check * (num_rounds+2) 
                        Else, # detectors = num_(basis)check * (num_rounds+2) + num_(other basis)check * num_rounds
    :return observable_flips: 2-dim numpy array of shape (num_trials, number of logical qubits). Each entry is a bit (0 or 1) that represents whether the logical Z flipped.
    '''

    if seed >= 0: 
        sampler = circuit.compile_detector_sampler(seed=seed)
    else:
        sampler = circuit.compile_detector_sampler()
    
    detection_events, observable_flips = sampler.sample(shots=num_trials, separate_observables=True)
    return detection_events, observable_flips


def get_codecap_pL(code, p, num_trials, decoder, dict, basis='Z', seed=-1, tqdm_on=False):

    if seed >= 0:
        np.random.seed(seed)
    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)

    parity_check_matrix = code.hz if basis == 'Z' else code.hx
    logical_codewords = code.lz if basis == 'Z' else code.lx

    bpd = decoder(parity_check_matrix, **dict)

    num_errors=0
    for i in iterator:
        noise = np.random.binomial(1, p, parity_check_matrix.shape[1])
        syndrome= parity_check_matrix @ noise % 2
        decoded_error = bpd.decode(syndrome)
        residual_error = (decoded_error + noise) %2
        if (logical_codewords @ residual_error % 2).any():
            num_errors += 1
    
    return num_errors/num_trials