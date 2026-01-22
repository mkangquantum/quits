"""
@author: Yingjia Lin
"""

from typing import Dict, FrozenSet, List

import numpy as np
from scipy.sparse import csc_matrix
import stim


############################Detector error matrix conversion codes##############################################################
'''
This part of the code is to convert a stim.DetectorErrorModel into a detector error matrix.
It is inspired by and adapted from:
1. BeliefMatching package (by Oscar Higgott): https://github.com/oscarhiggott/BeliefMatching/tree/main
2. Source codes for the paper "Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise":
    https://github.com/gongaa/SlidingWindowDecoder
    (by Anqi Gong)

'''

################################################################################################################################


def dict_to_csc_matrix_column_row(elements_dict, shape):
    '''
    Convert a dictionary into a `scipy.sparse.csc_matrix` with all the elements in this matrix are 1

    :params elements_dict: key: column indices, value: row indices
    :params shape: the shape of the resulting matrix

    :return a `scipy.sparse.csc_matrix` with column and row indices from the input dictionary. All elements are 1.
    '''
    # the non-zero elements in the matrix
    number_of_ones = sum(len(v) for v in elements_dict.values())
    data = np.ones(number_of_ones, dtype=np.uint8)
    # indices of the elements
    row_ind = np.zeros(number_of_ones, dtype=np.int64)
    col_ind = np.zeros(number_of_ones, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def dict_to_csc_matrix_row_column(elements_dict, shape):
    '''
    Convert a dictionary into a `scipy.sparse.csc_matrix` with all the elements in this matrix are 1

    :params elements_dict: key: row indices, value: column indices
    :params shape: the shape of the resulting matrix

    :return a `scipy.sparse.csc_matrix` with row and column indices from the input dictionary. All elements are 1.
    '''
    # Non-zero elements of the matrix
    number_of_ones = sum(len(v) for v in elements_dict.keys())
    data = np.ones(number_of_ones, dtype=np.uint8)
    # indices of the elements
    row_ind = np.zeros(number_of_ones, dtype=np.int64)
    col_ind = np.zeros(number_of_ones, dtype=np.int64)
    i = 0
    for v, col in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def detector_error_model_to_matrix(dem: stim.DetectorErrorModel):
    '''
    Obtain the detector error matrix from stim.DetectorErrorModel

    :param dem: Detector error model of the syndrome extraction circuit generated from stim

    :return check_matrix: detector error matrix. Each column represents a fault mechanism and each row represents a detector
    :return observables_matrix: the corresponding observable flips to each fault
    :return priors: the probability of each fault
    '''

    dem_dict: Dict[FrozenSet[int], int] = {}  # dictionary representation of detector error matrix, key: detector flips, value: fault_ids
    Logical_dict: Dict[int, FrozenSet[int]] = {}  # dictionary representation of logical observable flips, key: fault_ids, value: observable flips
    priors: List[float] = []  # error mechanism

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)

        if dets not in dem_dict:  # for syndrome not added to the dem_dict
            dem_dict[dets] = len(dem_dict)  # key: detector flips, value: fault_ids
            priors.append(prob)  # list of probability
            Logical_dict[dem_dict[dets]] = obs  # key: fault_ids, value: observable flips
        else:
            syndrome_id = dem_dict[dets]  # get the syndrome id when the syndrome is already added into the dem_dict
            priors[syndrome_id] = priors[syndrome_id] * (1 - prob) + prob * (1 - priors[syndrome_id])  # combining the probability

    for instruction in dem.flattened():

        if instruction.type == "error":  # fault mechanism in detector error model

            dets: List[int] = []
            obs: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)
            if len(dets) == 0:
                print(instruction)
            handle_error(p, dets, obs)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    detector_error_matrix = dict_to_csc_matrix_row_column(dem_dict,
                                      shape=(dem.num_detectors, len(dem_dict)))
    observables_matrix = dict_to_csc_matrix_column_row(Logical_dict, shape=(dem.num_observables, len(dem_dict)))
    priors = np.array(priors)
    return detector_error_matrix, observables_matrix, priors

############################################################################################################################################

############################Sliding window implementation with spacetime codes##############################################################


def spacetime(circuit, hz, W, F, num_cor_rounds):
    '''
    Obtain the spacetime slices of detector error matrix for sliding window decoder

    :param circuit: stim circuit
    :param hz: X or Z error parity check matrix of the code.
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param num_cor_rounds: number of windows before the last window

    :return window_check_set: a set of sliced spacetime detector error matrix for each window in sliding window decoder
    :return window_observable_set: a set of sliced observable matrix that marks the observable flips of each fault in each window
    :return window_priors_set: a set of probability for faults in each window
    :return window_update: the detector information update for next window of each fault mechanism in each window
    '''
    if F == 0:
        raise ValueError("Input parameter F cannot be zero.")
    model = circuit.detector_error_model(decompose_errors=False)  # detector error model of the circuit
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
    window_check_set = []
    window_observable_set = []
    window_priors_set = []
    window_update = []
    col_min = 0
    '''Check_matrix for each window'''
    for k in range(num_cor_rounds):
        window_check_matrix = check_matrix[k * F * hz.shape[0]:(k * F + W) * hz.shape[0], col_min:]
        if len(window_check_matrix.indptr) == 1:
            raise ValueError("There is no noise in one of the decoding window. This means there are redundant detectors that do not check for any error.")
        col_max = np.max(np.where(np.diff(window_check_matrix.indptr) > 0)[0])  # all the columns that affect the window
        window_check_matrix = window_check_matrix[:, :col_max + 1]
        window_check_set.append(window_check_matrix)

        '''corresponding flips of observables: only care about the part we fix'''
        F_correction = window_check_matrix[:F * hz.shape[0], :]
        cor_max = np.max(np.where(np.diff(F_correction.indptr) > 0)[0])
        window_observable_matrix = observable_matrix[:, col_min:cor_max + 1 + col_min]
        window_observable_set.append(window_observable_matrix)

        '''probability of each fault'''
        window_priors = priors[col_min:col_max + 1 + col_min]
        window_priors_set.append(window_priors)
        '''updating the detector flips for the next window'''
        updated_info = check_matrix[(k + 1) * F * hz.shape[0]:((k + 1) * F + 1) * hz.shape[0], col_min:cor_max + 1 + col_min]
        col_min = (cor_max + 1) + col_min
        window_update.append(updated_info)
    '''last window check matrix'''
    last_window_check_matrix = check_matrix[F * num_cor_rounds * hz.shape[0]:, col_min:]
    window_check_set.append(last_window_check_matrix)
    '''last window observable flip'''
    last_window_observable_matrix = observable_matrix[:, col_min:]
    window_observable_set.append(last_window_observable_matrix)
    '''last window prior'''
    last_window_priors = priors[col_min:]
    window_priors_set.append(last_window_priors)

    return window_check_set, window_observable_set, window_priors_set, window_update


__all__ = [
    "dict_to_csc_matrix_column_row",
    "dict_to_csc_matrix_row_column",
    "detector_error_model_to_matrix",
    "spacetime",
]
