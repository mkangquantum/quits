"""
@author: Yingjia Lin
"""

from ldpc.bplsd_decoder import BpLsdDecoder

from .sliding_window import sliding_window_phenom_mem, sliding_window_circuit_mem


def sliding_window_bplsd_phenom_mem(zcheck_samples, hz, lz, W, F, error_rate: float, max_iter=2, lsd_order=0, bp_method='product_sum', schedule='serial', lsd_method='lsd_cs', tqdm_on=False):
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
    # parameters of decoders
    dict1 = {'bp_method': bp_method,
            'max_iter': max_iter,
            'schedule': schedule,
            'lsd_method': lsd_method,
            'lsd_order': lsd_order,
          'error_rate': float(error_rate)}
    dict2 = {'bp_method': bp_method,
            'max_iter': max_iter,
            'schedule': schedule,
            'lsd_method': lsd_method,
            'lsd_order': lsd_order,
          'error_rate': float(error_rate)}
    logical_pred = sliding_window_phenom_mem(zcheck_samples, hz, lz, W, F, BpLsdDecoder, BpLsdDecoder, dict1, dict2, 'decode', 'decode', tqdm_on=tqdm_on)
    return logical_pred


def sliding_window_bplsd_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, max_iter=2, lsd_order=0, bp_method='product_sum', schedule='serial', lsd_method='lsd_cs', tqdm_on=False):
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
    # parameters of decoders
    dict1 = {'bp_method': bp_method,
            'max_iter': max_iter,
            'schedule': schedule,
            'lsd_method': lsd_method,
            'lsd_order': lsd_order}
    dict2 = {'bp_method': bp_method,
            'max_iter': max_iter,
            'schedule': schedule,
            'lsd_method': lsd_method,
            'lsd_order': lsd_order}
    logical_pred = sliding_window_circuit_mem(zcheck_samples, circuit, hz, lz, W, F, BpLsdDecoder, BpLsdDecoder, dict1, dict2, 'channel_probs', 'channel_probs', 'decode', 'decode', tqdm_on=tqdm_on)

    return logical_pred


__all__ = ["sliding_window_bplsd_phenom_mem", "sliding_window_bplsd_circuit_mem"]
