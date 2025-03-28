"""
@author: Yingjia Lin
"""

import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from tqdm import tqdm
from scipy.sparse import csc_matrix
import stim
from typing import List, FrozenSet, Dict
import warnings

def sliding_window_phenom_mem(zcheck_samples, hz, lz, W, F, decoder1, decoder2, dict1:dict, dict2:dict, function_name1:str, function_name2:str, tqdm_on=False):
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
    :param error_rate: Estimate of error rate in the context of code-capacity/phenomenological level simulations. 
                For circuit level, we suggest p * (num_layers + 3), where p is the depolarizing error rate 
                and num_layers is the circuit depth for each stabilizer measurement round
                e.g. for hgp codes, num_layers == code.count_color('east') + code.count_color('north') + code.count_color('south') + code.count_color('west')

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''
    if F == 0:
        raise ValueError("Input parameter F cannot be zero.")
    num_trials = zcheck_samples.shape[0]
    num_rounds = zcheck_samples.shape[1] // hz.shape[0] - 2

    #update the total number of windows for decoding, the size of the last window
    if 2+num_rounds-W>=0:
        num_cor_rounds=(2+num_rounds-W)//F#num_cor_rounds=num of windows before the last window
        if (2+num_rounds-W)%F!=0:# we can slide one more window if the remaining rounds>W
            num_cor_rounds+=1
    else:
        num_cor_rounds=0
        warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
    W_last=num_rounds+2-F*num_cor_rounds
    
    #update the window matrix and the decoder
    B = np.eye(W, dtype=int)
    for i in range(1, W):
        B[i, i-1] = 1         
    hz_phenom = np.column_stack((np.kron(np.eye(W, dtype=int), hz), np.kron(B, np.eye(hz.shape[0], dtype=int))))
    
    decoder_each_window = decoder1(csc_matrix(hz_phenom), **dict1)
    
    B_last = np.eye(W_last, dtype=int)   
    for i in range(1, W_last):
        B_last[i, i-1] = 1 
    B_last=B_last[:,:W_last-1] #The last round in this window is ideal
    
    hz_last = np.column_stack((np.kron(np.eye(W_last, dtype=int), hz), np.kron(B_last, np.eye(hz.shape[0], dtype=int))))
    decoder_last_window = decoder2(csc_matrix(hz_last), **dict2)

    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)
    logical_z_pred = np.zeros((num_trials, lz.shape[0]), dtype=int)

    for i in iterator:#each sample decoding
        accumulated_correction=np.zeros(hz.shape[1], dtype=int)
        syn_update=np.zeros(hz.shape[0], dtype=int)

        for k in range(num_cor_rounds):
            diff_syndrome = (zcheck_samples[i, F*k*hz.shape[0]:(F*k+W)*hz.shape[0]].copy()) % 2
            diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2

            decoded_errors = getattr(decoder_each_window, function_name1)(diff_syndrome)
            correction = np.sum(decoded_errors[:F*hz.shape[1]].reshape(F, hz.shape[1]).copy(),axis=0) % 2
            
            syn_update = decoded_errors[W*hz.shape[1]+(F-1)*hz.shape[0]:W*hz.shape[1]+F*hz.shape[0]].copy() 
            accumulated_correction = (accumulated_correction + correction) % 2
            
        #In the last round we just correct the whole window
        diff_syndrome = (zcheck_samples[i, (F*num_cor_rounds)*hz.shape[0]:].copy()) % 2
        diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2
        
        decoded_errors = getattr(decoder_last_window, function_name2)(diff_syndrome)
        correction = np.sum(decoded_errors[:W_last*hz.shape[1]].reshape(W_last, hz.shape[1]),axis=0) % 2
        
        accumulated_correction = (accumulated_correction + correction) % 2
        logical_z_pred[i,:] = (lz @ accumulated_correction) % 2
        
    return logical_z_pred

def sliding_window_bposd_phenom_mem(zcheck_samples, hz, lz, W, F, error_rate:float, max_iter=2, osd_order=0, bp_method='product_sum', schedule='serial',osd_method='osd_cs',tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024) implemented with BP-OSD decoder
    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory. 

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param error_rate: Estimate of error rate in the context of code-capacity/phenomenological level simulations. 
                For circuit level, we suggest p * (num_layers + 3), where p is the depolarizing error rate 
                and num_layers is the circuit depth for each stabilizer measurement round
                e.g. for hgp codes, num_layers == code.count_color('east') + code.count_color('north') + code.count_color('south') + code.count_color('west')
    :param max_iter: Maximum number of iterations for BP
    :param osd_order: Osd search depth
    :param bp_method: BP method for BP_OSD. Choose from ‘product_sum’ or ‘minimum_sum’
    :param schedule: choose from 'serial' or 'parallel'
    :param osd_method: choose from:  'osd_e', 'osd_cs', 'osd_0'

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''
    #parameters of decoders
    dict1={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'osd_method' : osd_method,
            'osd_order' : osd_order,
          'error_rate' : float(error_rate)}
    dict2={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'osd_method' : osd_method,
            'osd_order' : osd_order,
          'error_rate' : float(error_rate)}
    logical_pred = sliding_window_phenom_mem(zcheck_samples, hz, lz, W, F, BpOsdDecoder, BpOsdDecoder, dict1, dict2, 'decode', 'decode', tqdm_on=tqdm_on)
    return logical_pred

def sliding_window_bplsd_phenom_mem(zcheck_samples, hz, lz, W, F, error_rate:float, max_iter=2, lsd_order=0, bp_method='product_sum', schedule='serial',lsd_method='lsd_cs',tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024) implemented with BP-LSD decoder
    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory. 

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param error_rate: Estimate of error rate in the context of code-capacity/phenomenological level simulations. 
                For circuit level, we suggest p * (num_layers + 3), where p is the depolarizing error rate 
                and num_layers is the circuit depth for each stabilizer measurement round
                e.g. for hgp codes, num_layers == code.count_color('east') + code.count_color('north') + code.count_color('south') + code.count_color('west')
    :param max_iter: Maximum number of iterations for BP
    :param lsd_order: Lsd search depth
    :param bp_method: BP method for BP_LSD. Choose from ‘product_sum’ or ‘minimum_sum’
    :param schedule: choose from 'serial' or 'parallel'
    :param lsd_method: choose from:  'lsd_e', 'lsd_cs', 'lsd_0'

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''    
    #parameters of decoders
    dict1={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'lsd_method' : lsd_method,
            'lsd_order' : lsd_order,
          'error_rate' : float(error_rate)}
    dict2={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'lsd_method' : lsd_method,
            'lsd_order' : lsd_order,
          'error_rate' : float(error_rate)}
    logical_pred = sliding_window_phenom_mem(zcheck_samples, hz, lz, W, F, BpLsdDecoder, BpLsdDecoder, dict1, dict2, 'decode', 'decode', tqdm_on=tqdm_on)
    return logical_pred

############################Sliding window implementation with spacetime codes##############################################################

def dict_to_csc_matrix_column_row(elements_dict, shape):
    '''
    Convert a dictionary into a `scipy.sparse.csc_matrix` with all the elements in this matrix are 1
    
    :params elements_dict: key: column indices, value: row indices
    :params shape: the shape of the resulting matrix
    
    :return a `scipy.sparse.csc_matrix` with column and row indices from the input dictionary. All elements are 1.
    '''
    #the non-zero elements in the matrix
    number_of_ones = sum(len(v) for v in elements_dict.values())
    data = np.ones(number_of_ones, dtype=np.uint8)
    #indices of the elements
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
    #indices of the elements
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

    dem_dict: Dict[FrozenSet[int], int] = {} # dictionary representation of detector error matrix, key: detector flips, value: fault_ids
    Logical_dict: Dict[int, FrozenSet[int]] = {} # dictionary representation of logical observable flips, key: fault_ids, value: observable flips
    priors: List[float] = [] # error mechanism

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        
        if dets not in dem_dict:#for syndrome not added to the dem_dict            
            dem_dict[dets] = len(dem_dict)#key: detector flips, value: fault_ids
            priors.append(prob)#list of probability
            Logical_dict[dem_dict[dets]] = obs#key: fault_ids, value: observable flips
        else:
            syndrome_id = dem_dict[dets]# get the syndrome id when the syndrome is already added into the dem_dict
            priors[syndrome_id] = priors[syndrome_id]*(1-prob)+prob*(1-priors[syndrome_id])#combining the probability
            
    for instruction in dem.flattened():
        
        if instruction.type == "error":# fault mechanism in detector error model
            
            dets: List[int] = []
            obs: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)
            if len(dets)==0:
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
    priors=np.array(priors)
    return detector_error_matrix, observables_matrix, priors

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
    model = circuit.detector_error_model(decompose_errors=False)# detector error model of the circuit
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
    window_check_set=[]
    window_observable_set=[]
    window_priors_set=[]
    window_update=[]
    col_min=0
    '''Check_matrix for each window'''
    for k in range(num_cor_rounds):
        window_check_matrix=check_matrix[k*F*hz.shape[0]:(k*F+W)*hz.shape[0], col_min:]
        col_max=np.max(np.where(np.diff(window_check_matrix.indptr) > 0)[0]) #all the columns that affect the window
        window_check_matrix=window_check_matrix[:,:col_max+1]
        window_check_set.append(window_check_matrix)
        
        '''corresponding flips of observables: only care about the part we fix'''
        F_correction=window_check_matrix[:F*hz.shape[0],:]
        cor_max=np.max(np.where(np.diff(F_correction.indptr) > 0)[0])
        window_observable_matrix=observable_matrix[:,col_min:cor_max+1+col_min]
        window_observable_set.append(window_observable_matrix)
        
        
        '''probability of each fault'''
        window_priors=priors[col_min:col_max+1+col_min]
        window_priors_set.append(window_priors)
        '''updating the detector flips for the next window'''
        updated_info=check_matrix[(k+1)*F*hz.shape[0]:((k+1)*F+1)*hz.shape[0],col_min:cor_max+1+col_min]
        col_min=(cor_max+1)+col_min
        window_update.append(updated_info)
    '''last window check matrix'''
    last_window_check_matrix=check_matrix[F*num_cor_rounds*hz.shape[0]:,col_min:]
    window_check_set.append(last_window_check_matrix)
    '''last window observable flip'''
    last_window_observable_matrix=observable_matrix[:,col_min:]
    window_observable_set.append(last_window_observable_matrix)
    '''last window prior'''
    last_window_priors=priors[col_min:]
    window_priors_set.append(last_window_priors)
    
    return window_check_set,window_observable_set,window_priors_set,window_update    
    
def sliding_window_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, decoder1, decoder2, dict1:dict, dict2:dict,
                               error_rate_name1:str, error_rate_name2:str,
                               function_name1:str, function_name2:str, tqdm_on=False):
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
    
    #update the total number of windows for decoding, the size of the last window
    if 2+num_rounds-W>=0:
        num_cor_rounds=(2+num_rounds-W)//F#num_cor_rounds=num of windows before the last window
        if (2+num_rounds-W)%F!=0:# we can slide one more window if the remaining rounds>W
            num_cor_rounds+=1
    else:
        num_cor_rounds=0
        warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
    W_last=num_rounds+2-F*num_cor_rounds
    #update the window matrix and the decoder
    #spacetime detector error matrix
    window_check_set,window_observable_set,window_priors_set,window_update = spacetime(circuit,hz, W, F,num_cor_rounds)
    #decoder for each window
    decoder=[]
    for i in range(len(window_check_set)-1):
        dict1[error_rate_name1]=window_priors_set[i]
        decoder_each_window = decoder1(window_check_set[i], **dict1)
        decoder.append(decoder_each_window)
    dict2[error_rate_name2]=window_priors_set[len(window_check_set)-1]
    decoder_each_window=decoder2(window_check_set[len(window_check_set)-1],**dict2)
    decoder.append(decoder_each_window)
        
    #start decoding
    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)
    logical_z_pred = np.zeros((num_trials, lz.shape[0]), dtype=int)
    
    for i in iterator:#each sample decoding
        accumulated_correction=np.zeros(window_observable_set[0].shape[0], dtype=int)
        syn_update=np.zeros(hz.shape[0], dtype=int)

        for k in range(num_cor_rounds):
            #syndrome of the window
            diff_syndrome = (zcheck_samples[i, F*k*hz.shape[0]:(F*k+W)*hz.shape[0]].copy()) % 2
            diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2#update the syndrome based on the previous window decoding
            
            
            decoded_errors = getattr(decoder[k], function_name1)(diff_syndrome)
            correction=window_observable_set[k]@decoded_errors[:window_observable_set[k].shape[1]]%2#interpret the correction operation as final observable flips
            
            syn_update = window_update[k]@decoded_errors[:window_observable_set[k].shape[1]]%2
            accumulated_correction = (accumulated_correction + correction) % 2
            
        #In the last round we just correct the whole window
        #syndrome of last round
        diff_syndrome = (zcheck_samples[i, (F*num_cor_rounds)*hz.shape[0]:].copy()) % 2
        diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2
        #Observable flips based on correction
        decoded_errors = getattr(decoder[num_cor_rounds], function_name2)(diff_syndrome)
        correction=window_observable_set[num_cor_rounds]@decoded_errors%2
        accumulated_correction = (accumulated_correction + correction) % 2
        #Predicted observable flips
        logical_z_pred[i,:] = accumulated_correction
        
    return logical_z_pred

def sliding_window_bposd_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, max_iter=2, osd_order=0, bp_method='product_sum', schedule='serial',osd_method='osd_cs',tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024) implemented with BP-OSD decoder and spacetime detector error matrix
    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory. 

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param circuit: syndrome extraction circuit
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param max_iter: Maximum number of iterations for BP
    :param osd_order: Osd search depth
    :param bp_method: BP method for BP_OSD. Choose from ‘product_sum’ or ‘minimum_sum’
    :param schedule: choose from 'serial' or 'parallel'
    :param osd_method: choose from:  'osd_e', 'osd_cs', 'osd_0'

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''
    #parameters of decoders
    dict1={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'osd_method' : osd_method,
            'osd_order' : osd_order}
    dict2={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'osd_method' : osd_method,
            'osd_order' : osd_order}
    logical_pred = sliding_window_circuit_mem(zcheck_samples, circuit, hz, lz, W, F,BpOsdDecoder, BpOsdDecoder, dict1, dict2, 'channel_probs','channel_probs', 'decode', 'decode', tqdm_on=tqdm_on)
    
    return logical_pred

def sliding_window_bplsd_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, max_iter=2, lsd_order=0, bp_method='product_sum', schedule='serial',lsd_method='lsd_cs',tqdm_on=False):
    '''
    Sliding window decoder in S. Huang and S. Puri, PRA 110, 012453 (2024) implemented with BP-LSD decoder and spacetime detector error matrix
    For convenience the notation assumes z-type memory, but the code works equivalently for x-type memory. 

    :param zcheck_samples: 2-dim numpy array of detector results; see get_stim_Zmem_result in simulation.py. Shape (# trials, # Z-check qubits * (# rounds+1))
    :param circuit: syndrome extraction circuit
    :param hz: Parity check matrix (in the code-capacity level) representing the Z stabilizers of the qec code. Shape ((# Z-check qubits, # data qubits))
    :param lz: Logical codeword matrix of the qec code. Shape ((# logical qubits, # data qubits))
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param max_iter: Maximum number of iterations for BP
    :param lsd_order: Lsd search depth
    :param bp_method: BP method for BP_LSD. Choose from ‘product_sum’ or ‘minimum_sum’
    :param schedule: choose from 'serial' or 'parallel'
    :param lsd_method: choose from:  'lsd_e', 'lsd_cs', 'lsd_0'

    :return logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)
    '''
    #parameters of decoders
    dict1={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'lsd_method' : lsd_method,
            'lsd_order' : lsd_order}
    dict2={'bp_method' : bp_method,
            'max_iter' : max_iter,
            'schedule' : schedule,
            'lsd_method' : lsd_method,
            'lsd_order' : lsd_order}
    logical_pred = sliding_window_circuit_mem(zcheck_samples, circuit, hz, lz, W, F,BpLsdDecoder, BpLsdDecoder, dict1, dict2, 'channel_probs','channel_probs', 'decode', 'decode', tqdm_on=tqdm_on)
    
    return logical_pred
