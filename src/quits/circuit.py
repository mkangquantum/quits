"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np

from .noise import ErrorModel

def check_overlapping_CX(circuit, verbose=True):
    """Return CX instructions that reuse a qubit within the same instruction."""
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
        """Initialize an empty circuit helper with default noise settings."""
        self.circuit = ''
        self.margin = ''
        self.all_qubits = all_qubits
        self.set_error_model(ErrorModel())
        
    def set_all_qubits(self, all_qubits):
        """Update the tracked set of qubits in the circuit."""
        self.all_qubits = all_qubits
    
    def set_error_model(self, error_model):
        """Set the error model used by gate and measurement helpers."""
        self.error_model = error_model
        self.idle_error = error_model.idle_error
        self.sqgate_error = error_model.sqgate_error
        self.tqgate_error = error_model.tqgate_error
        self.spam_error = error_model.spam_error

    def set_error_rates(self, idle_error, sqgate_error, tqgate_error, spam_error):
        """Set error rates by constructing and applying an ErrorModel."""
        self.set_error_model(ErrorModel(idle_error, sqgate_error, tqgate_error, spam_error))
        
    def start_loop(self, num_rounds):
        """Start a Stim REPEAT block and increase indentation margin."""
        c = 'REPEAT %d {\n'%num_rounds
        self.circuit += c
        self.margin = '    ' 
        return c
        
    def end_loop(self):
        """Close a Stim REPEAT block and reset indentation margin."""
        c = '}\n'
        self.circuit += c
        self.margin = ''
        return c
        
    def add_tick(self):
        """Append a TICK separator."""
        c = self.margin + 'TICK\n'
        self.circuit += c
        return c   
        
    def add_reset(self, qubits, basis='Z'):        
        """Append reset operations (R/RX) and optional SPAM noise."""
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
        """Append idling noise on the given qubits."""
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
        """Append H gates and optional single-qubit gate noise."""
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
        """Add a Hadamard layer with idling errors on other qubits and a TICK."""
        c1 = self.add_hadamard(qubits)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3
    
    def add_cnot(self, qubits):
        """Append CX gates and optional two-qubit gate noise."""
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
        """Add a CNOT layer with idling errors on other qubits and a TICK."""
        c1 = self.add_cnot(qubits)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3   


    def add_measure(self, qubits, basis='Z'):
        """Append M/MX operations with optional pre-measurement SPAM noise."""
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

    def add_measure_layer(self, qubits, basis='Z'):
        """Add a measurement layer with idling errors on other qubits and a TICK."""
        c1 = self.add_measure(qubits, basis)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3     
    
    def add_measure_reset(self, qubits, error_free_reset=False):       
        """Append MR operations with optional SPAM errors before/after reset."""
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
        """Add an MR layer with idling errors on other qubits and a TICK."""
        c1 = self.add_measure_reset(qubits, error_free_reset)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        c3 = self.add_tick()
        return c1 + c2 + c3  
    
    def add_detector(self, inds):
        """Append a DETECTOR instruction for the given rec offsets."""
        c = self.margin + 'DETECTOR '
        for ind in inds:
            c += 'rec[-%d] '%ind
        c += '\n'
        
        self.circuit += c
        
    def add_observable(self, observable_no, inds):
        """Append an OBSERVABLE_INCLUDE instruction."""
        c = self.margin + 'OBSERVABLE_INCLUDE(%d) '%observable_no
        for ind in inds:
            c += 'rec[-%d] '%ind
        c += '\n'
        
        self.circuit += c
        return c
    
