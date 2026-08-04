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

    basis = basis.upper()
    if basis == 'Z':
        parity_check_matrix = code.hz
        logical_codewords = code.lz
    elif basis == 'X':
        parity_check_matrix = code.hx
        logical_codewords = code.lx
    else:
        raise ValueError("basis must be 'Z' or 'X'")

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


def spacetime_matrices(circuit):
    """Spacetime check/logical pair ``(H, L)`` of a memory circuit's DEM.

    Columns are circuit fault mechanisms; rows of ``H`` are detector
    parities, rows of ``L`` the logical observables each mechanism flips
    (the spacetime representation used by arXiv:2607.28795 for hook-error
    screening). Applicable to any Stim circuit with detectors and
    observables.
    """
    dem = circuit.detector_error_model(flatten_loops=True)
    cols = []
    for inst in dem:
        if inst.type != "error":
            continue
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        cols.append((dets, obs))
    H = np.zeros((dem.num_detectors, len(cols)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(cols)), dtype=np.uint8)
    for j, (dets, obs) in enumerate(cols):
        for d in dets:
            H[d, j] ^= 1
        for o in obs:
            L[o, j] ^= 1
    return H, L


def estimate_spacetime_distance(H, L, *, num_trials=100, seed=0, osd_order=2):
    """Randomized upper bound on ``min{|e| : He = 0, Le ≠ 0}`` via BP-OSD.

    Each trial draws a random logical combination ``l`` (a row combination
    of ``L``), decodes the augmented system ``[H; l]·e = [0; 1]`` with
    ``ldpc.BpOsdDecoder`` (OSD guarantees the syndrome is satisfied), and
    keeps the lightest fault pattern found. This is the circuit-level
    distance of the memory experiment for the sampled basis; like all
    randomized estimators it only ever over-estimates the true distance.
    """
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix, vstack

    rng = np.random.default_rng(seed)
    Hs = csr_matrix(H)
    k = L.shape[0]
    best = None
    for _ in range(num_trials):
        w = rng.integers(0, 2, size=k)
        if not w.any():
            w[rng.integers(k)] = 1
        l = ((w @ L) % 2).astype(np.uint8)
        if not l.any():
            continue
        Haug = vstack([Hs, csr_matrix(l.reshape(1, -1))]).tocsr()
        decoder = BpOsdDecoder(Haug, error_rate=0.01, max_iter=20,
                               bp_method="ms", osd_method="osd_cs",
                               osd_order=osd_order)
        syndrome = np.zeros(Haug.shape[0], dtype=np.uint8)
        syndrome[-1] = 1
        weight = int(decoder.decode(syndrome).sum())
        best = weight if best is None else min(best, weight)
    return best
