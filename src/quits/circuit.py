"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np

def get_qldpc_mem_circuit(code, idle_error, sqgate_error, tqgate_error, spam_error, num_rounds,\
                          basis='Z', get_all_detectors=False, noisy_init=True, noisy_meas=False):
    '''
    Returns the full Stim circuit for circuit-level simulations.
    Errors occur at each and every gate. 

    :param code: The qldpc_code class from qldpc.py
    :param idle_error: Rate of depolarizing error. Float or tuple of 3 floats (px, py, pz)
    :param sqgate_error: Rate of single-qubit gate error. Float or tuple of 3 floats (px, py, pz)
    :param tqgate_error: Rate of two-qubit gate error. Float or tuple of 15 floats (pix, ..., pzz)
    :param spam_error: Rate of state preparation and measurement error. Float
    :param num_rounds: Number of stabilizer measurement rounds
    :param basis: Basis in which logical codewords are stored. Options are 'Z' and 'X'.
    :param get_all_detectors: If True, all detectors in the Z and X bases are obtained. 
                              If False, only the detectors in the memory basis are obtained. 
    :param noisy_init: If False, reset of data and syndrome qubits at the 0-th round is assumed to be error free.
    :param noisy_meas: If False, final measurement of data qubits is assumed to be error free
    :return circuit: String that can be converted to Stim circuit by stim.Circuit()
    '''
    
    basis = basis.upper()
    if basis not in ('Z', 'X'):
        raise ValueError("basis must be 'Z' or 'X'")
    get_Z_detectors = True if basis == 'Z' or get_all_detectors else False
    get_X_detectors = True if basis == 'X' or get_all_detectors else False
    directions = list(code.direction_inds.keys())
    circ = Circuit(code.all_qubits)
    
    def _add_stabilizer_round():
        circ.add_hadamard_layer(code.xcheck_qubits)
        for direction_ind in range(len(directions)):
            direction = directions[direction_ind]
            for color in range(code.num_colors[direction]):
                edges = code.colored_edges[direction_ind][color]
                circ.add_cnot_layer(edges)
        circ.add_hadamard_layer(code.xcheck_qubits)
        circ.add_measure_reset_layer(code.check_qubits)
        return
    
    ################## Logical state prep ##################
    if noisy_init:
        circ.set_error_rates(idle_error, sqgate_error, tqgate_error, spam_error)
    else:
        circ.set_error_rates(0., 0., 0., 0.)
    circ.add_reset(code.data_qubits, basis)
    circ.add_reset(code.check_qubits)
    circ.add_tick()

    circ.set_error_rates(idle_error, sqgate_error, tqgate_error, spam_error)
    _add_stabilizer_round()

    if basis == 'Z':
        for i in range(1, len(code.zcheck_qubits)+1)[::-1]:     
            circ.add_detector([len(code.xcheck_qubits) + i])
    elif basis == 'X':
        for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
            circ.add_detector([i])   
    
    ############## Logical memory w/ noise ###############
    circ.set_error_rates(idle_error, sqgate_error, tqgate_error, spam_error)
    
    if num_rounds > 0: 
        circ.start_loop(num_rounds)  
        
        _add_stabilizer_round()
        
        if get_Z_detectors:
            for i in range(1, len(code.zcheck_qubits)+1)[::-1]:
                ind = len(code.xcheck_qubits) + i        
                circ.add_detector([ind, ind + len(code.check_qubits)])
        if get_X_detectors:
            for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
                circ.add_detector([i, i + len(code.check_qubits)])
            
        circ.end_loop()
        
    ################## Logical measurement ##################   
    if not noisy_meas:
        circ.set_error_rates(0., 0., 0., 0.)
        
    circ.add_measure(code.data_qubits, basis)
    
    if basis == 'Z':
        for i in range(1, len(code.zcheck_qubits)+1)[::-1]:
            inds = np.array([len(code.data_qubits) + len(code.xcheck_qubits) + i])
            inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hz[len(code.zcheck_qubits)-i, :]==1)[0]))
            circ.add_detector(inds)
        
        for i in range(len(code.lz)):
            circ.add_observable(i, len(code.data_qubits) - np.where(code.lz[i,:]==1)[0])
            
    elif basis == 'X':
        for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
            inds = np.array([len(code.data_qubits) + i])
            inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hx[len(code.xcheck_qubits)-i, :]==1)[0]))
            circ.add_detector(inds)
        
        for i in range(len(code.lx)):
            circ.add_observable(i, len(code.data_qubits) - np.where(code.lx[i,:]==1)[0])     
            
    return circ.circuit

def check_overlapping_CX(circuit, verbose=True):
    '''
    Check for overlapping CX gates in the same layer of a stim circuit.

    :param circuit: stim.Circuit
    :param verbose: If True, print overlaps as they are found.
    :return: List of (index, duplicates) for each offending CX instruction.
    '''
    overlaps = []
    for i in range(len(circuit)):
        if circuit[i].name == 'CX':
            size = len(circuit[i].targets_copy())
            gate_list = np.zeros(size, dtype=int)
            for j in range(size):
                gate_list[j] = circuit[i].targets_copy()[j].qubit_value
            unique, counts = np.unique(gate_list, return_counts=True)

            duplicates = unique[counts > 1]
            if duplicates.size > 0:
                if verbose:
                    print("Duplicates found:", i, duplicates)
                overlaps.append((i, duplicates.copy()))
    if verbose and not overlaps:
        print("No overlapping CX gates found.")
    return overlaps


class Circuit:
    '''
    Class containing helper functions for writing Stim circuits (https://github.com/quantumlib/Stim)
    '''
    
    def __init__(self, all_qubits):
        
        self.circuit = ''
        self.margin = ''
        self.all_qubits = all_qubits
        self.idle_error = 0.
        self.sqgate_error = 0.
        self.tqgate_error = 0.
        self.spam_error = 0.
        
    def set_all_qubits(self, all_qubits):
        self.all_qubits = all_qubits
    
    def set_error_rates(self, idle_error, sqgate_error, tqgate_error, spam_error):
        self.idle_error = idle_error
        self.sqgate_error = sqgate_error
        self.tqgate_error = tqgate_error
        self.spam_error = spam_error       
        
    def start_loop(self, num_rounds):
        c = 'REPEAT %d {\n'%num_rounds
        self.circuit += c
        self.margin = '    ' 
        return c
        
    def end_loop(self):
        c = '}\n'
        self.circuit += c
        self.margin = ''
        return c
        
    def add_tick(self):
        c = self.margin + 'TICK\n'
        self.circuit += c
        return c   
        
    def add_reset(self, qubits, basis='Z'):        
        basis = basis.upper()
        if basis not in ('Z', 'X'):
            raise ValueError("basis must be 'Z' or 'X'")
        
        c = self.margin
        if basis == 'Z':
            c += 'R '
        elif basis == 'X':
            c += 'RX '
        for q in qubits:
            c += '%d '%q
        c += '\n'
        
        if self.spam_error > 0.:
            c += self.margin
            if basis == 'Z':
                c += 'X_ERROR(%.10f) '%self.spam_error
            elif basis == 'X':
                c += 'Z_ERROR(%.10f) '%self.spam_error            
            for q in qubits:
                c += '%d '%q            
            c += '\n'
        
        self.circuit += c
        return c       
    
    def add_idle(self, qubits):
        if self.idle_error == 0.:
            return ''
        
        c = self.margin
        if type(self.idle_error) == float or type(self.idle_error) == np.float64:
            c += 'DEPOLARIZE1(%.10f) '%self.idle_error
        else:
            c += 'PAULI_CHANNEL_1('
            for p in self.idle_error:
                c += '%.10f, '%p
            c = c[:-2] + ') '
        for q in qubits:
            c += '%d '%q
        c += '\n'
        
        self.circuit += c
        return c
    
    def add_hadamard(self, qubits):
        c = self.margin
        c += 'H '
        for q in qubits:
            c += '%d '%q
        c += '\n'
        
        if self.sqgate_error != 0. and self.sqgate_error != 0:
            c += self.margin
            if type(self.sqgate_error) == float or type(self.sqgate_error) == np.float64:
                c += 'DEPOLARIZE1(%.10f) '%self.sqgate_error
            else:
                c += 'PAULI_CHANNEL_1('
                for p in self.sqgate_error:
                    c += '%.10f, '%p
                c = c[:-2] + ') '
            for q in qubits:
                c += '%d '%q
            c += '\n'
            
        self.circuit += c
        return c
    
    def add_hadamard_layer(self, qubits):
        c1 = self.add_hadamard(qubits)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3
    
    def add_cnot(self, qubits):
        c = self.margin
        c += 'CX '
        for q in qubits:
            c += '%d '%q
        c += '\n'
        
        if self.tqgate_error != 0. and self.tqgate_error != 0:
            c += self.margin
            if type(self.tqgate_error) == float or type(self.tqgate_error) == np.float64:
                c += 'DEPOLARIZE2(%.10f) '%self.tqgate_error
            else:
                c += 'PAULI_CHANNEL_2('
                for p in self.tqgate_error:
                    c += '%.10f, '%p
                c = c[:-2] + ') '
            for q in qubits:
                c += '%d '%q
            c += '\n'
            
        self.circuit += c
        return c        
        
    def add_cnot_layer(self, qubits):
        c1 = self.add_cnot(qubits)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3    
    
    def add_measure_reset(self, qubits, error_free_reset=False):       
        c = ''
        if self.spam_error > 0.:
            c += self.margin
            c += 'X_ERROR(%.10f) '%self.spam_error           
            for q in qubits:
                c += '%d '%q            
            c += '\n'
            
        c += self.margin
        c += 'MR '
        for q in qubits:
            c += '%d '%q
        c += '\n'   
        
        if self.spam_error > 0. and not error_free_reset:
            c += self.margin
            c += 'X_ERROR(%.10f) '%self.spam_error          
            for q in qubits:
                c += '%d '%q            
            c += '\n'        
            
        self.circuit += c
        return c
    
    def add_measure_reset_layer(self, qubits, error_free_reset=False):
        c1 = self.add_measure_reset(qubits, error_free_reset)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3  
        
    def add_measure(self, qubits, basis='Z'):
        basis = basis.upper()
        if basis not in ('Z', 'X'):
            raise ValueError("basis must be 'Z' or 'X'")
        
        c = ''
        if self.spam_error > 0.:
            c += self.margin
            if basis == 'Z':
                c += 'X_ERROR(%.10f) '%self.spam_error
            elif basis == 'X':
                c += 'Z_ERROR(%.10f) '%self.spam_error            
            for q in qubits:
                c += '%d '%q            
            c += '\n'
            
        c += self.margin
        if basis == 'Z':
            c += 'M '
        elif basis == 'X':
            c += 'MX '
        for q in qubits:
            c += '%d '%q
        c += '\n'        
        
        self.circuit += c
        return c 
    
    def add_detector(self, inds):
        c = self.margin + 'DETECTOR '
        for ind in inds:
            c += 'rec[-%d] '%ind
        c += '\n'
        
        self.circuit += c
        
    def add_observable(self, observable_no, inds):
        c = self.margin + 'OBSERVABLE_INCLUDE(%d) '%observable_no
        for ind in inds:
            c += 'rec[-%d] '%ind
        c += '\n'
        
        self.circuit += c
        return c
    
