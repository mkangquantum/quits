"""
Bivariate bicycle (BB) code construction.
"""

import numpy as np
import stim

from ..circuit import Circuit
from ..gf2_util import compute_lz_and_lx
from ..noise import ErrorModel
from .circuit_construction import CircuitBuildOptions
from .base import QldpcCode


class BbCode(QldpcCode):
    def __init__(self, l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows):
        """
        :param l: Size of the x-shift (first cyclic dimension).
        :param m: Size of the y-shift (second cyclic dimension).
        :param A_x_pows: Powers for x terms in A(x, y).
        :param A_y_pows: Powers for y terms in A(x, y).
        :param B_x_pows: Powers for x terms in B(x, y).
        :param B_y_pows: Powers for y terms in B(x, y).
        """
        # Reference: bivariate bicycle codes (BB). See the original BB construction papers.
        super().__init__()

        if l <= 0 or m <= 0:
            raise ValueError("l and m must be positive integers.")

        self.l, self.m = l, m 
        self.A_x_pows, self.A_y_pows = A_x_pows, A_y_pows
        self.B_x_pows, self.B_y_pows = B_x_pows, B_y_pows

        S_l = self.get_circulant_mat(self.l, -1)
        S_m = self.get_circulant_mat(self.m, -1)

        x = np.kron(S_l, np.eye(self.m, dtype=int))
        y = np.kron(np.eye(self.l, dtype=int), S_m)

        size = self.l * self.m
        A = np.zeros((size, size), dtype=int)
        for power in self.A_x_pows:
            A += np.linalg.matrix_power(x, power)
        for power in self.A_y_pows:
            A += np.linalg.matrix_power(y, power)

        B = np.zeros((size, size), dtype=int)
        for power in self.B_y_pows:
            B += np.linalg.matrix_power(y, power)
        for power in self.B_x_pows:
            B += np.linalg.matrix_power(x, power)

        self.A = (A % 2).astype(np.uint8)
        self.B = (B % 2).astype(np.uint8)

        self.hx = np.hstack((self.A, self.B))
        self.hz = np.hstack((self.B.T, self.A.T))
        self.lz, self.lx = compute_lz_and_lx(self.hz, self.hx)

    def build_circuit(
        self,
        strategy="freeform",
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        **opts,
    ):
        '''
        Build a circuit for this BB code using the selected construction strategy.

        :param strategy: Circuit-construction strategy name (e.g., "freeform").
        :param error_model: ErrorModel specifying idle/single-/two-qubit/SPAM noise.
        :param num_rounds: Number of noisy syndrome-extraction rounds after the zeroth round.
        :param basis: Logical storage/measurement basis, either "Z" or "X".
        :param circuit_build_options: CircuitBuildOptions controlling detector and noise toggles.
        :param opts: Additional keyword arguments for BB freeform construction details.
        :return: Stim circuit.
        '''
        if strategy != "freeform":
            return super().build_circuit(
                strategy=strategy,
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
                **opts,
            )
        return self._build_freeform_circuit(
            error_model=error_model,
            num_rounds=num_rounds,
            basis=basis,
            circuit_build_options=circuit_build_options,
            **opts,
        )

    def _build_freeform_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
    ):
        n = int(self.hx.shape[1])
        if n % 2 != 0:
            raise ValueError("Number of data qubits must be even.")
        half = n // 2

        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")

        basis = basis.upper()
        if basis not in ("Z", "X"):
            raise ValueError("basis must be 'Z' or 'X'.")
        get_Z_detectors = True if basis == 'Z' or circuit_build_options.get_all_detectors else False
        get_X_detectors = True if basis == 'X' or circuit_build_options.get_all_detectors else False

        S_l = self.get_circulant_mat(self.l, -1)
        S_m = self.get_circulant_mat(self.m, -1)
        x = np.kron(S_l, np.eye(self.m, dtype=int))
        y = np.kron(np.eye(self.l, dtype=int), S_m)

        A_list = [
            np.linalg.matrix_power(x, power) for power in self.A_x_pows
        ] + [
            np.linalg.matrix_power(y, power) for power in self.A_y_pows
        ]
        B_list = [
            np.linalg.matrix_power(y, power) for power in self.B_y_pows
        ] + [
            np.linalg.matrix_power(x, power) for power in self.B_x_pows
        ]

        if len(A_list) != 3 or len(B_list) != 3:
            raise ValueError("A and B must each define exactly 3 shift terms.")

        a1, a2, a3 = A_list
        b1, b2, b3 = B_list

        def nnz(m):
            rows, cols = np.nonzero(m)
            order = np.argsort(rows)
            return cols[order]

        A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
        B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)
        A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
        B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

        x_check_offset = 0
        l_data_offset = half
        r_data_offset = n
        z_check_offset = n + half

        data_qubits = list(np.arange(l_data_offset, l_data_offset + n, dtype=int))
        zcheck_qubits = list(np.arange(z_check_offset, z_check_offset + half, dtype=int))
        xcheck_qubits = list(np.arange(x_check_offset, x_check_offset + half, dtype=int))

        self.data_qubits = np.array(data_qubits)
        self.zcheck_qubits = np.array(zcheck_qubits)
        self.xcheck_qubits = np.array(xcheck_qubits)
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))

        circ = Circuit(self.all_qubits)

        ########### Code for adding syndrome extraction round ##############
        def make_edges(control_offset, target_offset, mapping, mapping_option):
            if mapping_option == 'c':
                return [(control_offset + int(mapping[i]), target_offset + i) for i in range(half)]
            elif mapping_option == 't':
                return [(control_offset + i, target_offset + int(mapping[i])) for i in range(half)]
            else:
                raise ValueError("mapping_option must be 'c' or 't'.")

        edges_round1 = make_edges(r_data_offset, z_check_offset, A1_T, 'c')
        edges_round2 = make_edges(x_check_offset, l_data_offset, A2, 't') + make_edges(
            r_data_offset, z_check_offset, A3_T, 'c'
        )
        edges_round3 = make_edges(x_check_offset, r_data_offset, B2, 't') + make_edges(
            l_data_offset, z_check_offset, B1_T, 'c'
        )
        edges_round4 = make_edges(x_check_offset, r_data_offset, B1, 't') + make_edges(
            l_data_offset, z_check_offset, B2_T, 'c'
        )
        edges_round5 = make_edges(x_check_offset, r_data_offset, B3, 't') + make_edges(
            l_data_offset, z_check_offset, B3_T, 'c'
        )
        edges_round6 = make_edges(x_check_offset, l_data_offset, A1, 't') + make_edges(
            r_data_offset, z_check_offset, A2_T, 'c'
        )
        edges_round7 = make_edges(x_check_offset, l_data_offset, A3, 't')

        def flatten(edges):
            return [q for edge in edges for q in edge]

        def _add_stabilizer_round():
            circ.add_hadamard_layer(self.xcheck_qubits)
            circ.add_cnot_layer(flatten(edges_round1))
            circ.add_cnot_layer(flatten(edges_round2))
            circ.add_cnot_layer(flatten(edges_round3))
            circ.add_cnot_layer(flatten(edges_round4))
            circ.add_cnot_layer(flatten(edges_round5))
            circ.add_cnot_layer(flatten(edges_round6))
            circ.add_cnot_layer(flatten(edges_round7))
            circ.add_hadamard_layer(self.xcheck_qubits)
            circ.add_measure_reset_layer(self.check_qubits)

        ################## Logical state prep ##################
        if circuit_build_options.noisy_zeroth_round:
            circ.set_error_model(error_model)
        else:
            circ.set_error_model(ErrorModel.zero())

        circ.add_reset(self.data_qubits, basis)
        circ.add_reset(self.check_qubits)
        circ.add_tick()

        _add_stabilizer_round()

        if basis == 'Z':
            for i in range(1, len(self.zcheck_qubits)+1)[::-1]:
                circ.add_detector([len(self.xcheck_qubits) + i])
        elif basis == 'X':
            for i in range(1, len(self.xcheck_qubits)+1)[::-1]:
                circ.add_detector([i])

        ############## Logical memory w/ noise ###############
        circ.set_error_model(error_model)

        if num_rounds > 0:
            circ.start_loop(num_rounds)

            _add_stabilizer_round()

            if get_Z_detectors:
                for i in range(1, len(self.zcheck_qubits)+1)[::-1]:
                    ind = len(self.xcheck_qubits) + i
                    circ.add_detector([ind, ind + len(self.check_qubits)])
            if get_X_detectors:
                for i in range(1, len(self.xcheck_qubits)+1)[::-1]:
                    circ.add_detector([i, i + len(self.check_qubits)])

            circ.end_loop()

        ################## Logical measurement ##################
        if not circuit_build_options.noisy_final_meas:
            circ.set_error_model(ErrorModel.zero())

        circ.add_measure(self.data_qubits, basis)

        if basis == 'Z':
            for i in range(1, len(self.zcheck_qubits)+1)[::-1]:
                inds = np.array([len(self.data_qubits) + len(self.xcheck_qubits) + i])
                inds = np.concatenate((inds, len(self.data_qubits) - np.where(self.hz[len(self.zcheck_qubits)-i, :]==1)[0]))
                circ.add_detector(inds)

            for i in range(len(self.lz)):
                circ.add_observable(i, len(self.data_qubits) - np.where(self.lz[i,:]==1)[0])

        elif basis == 'X':
            for i in range(1, len(self.xcheck_qubits)+1)[::-1]:
                inds = np.array([len(self.data_qubits) + i])
                inds = np.concatenate((inds, len(self.data_qubits) - np.where(self.hx[len(self.xcheck_qubits)-i, :]==1)[0]))
                circ.add_detector(inds)

            for i in range(len(self.lx)):
                circ.add_observable(i, len(self.data_qubits) - np.where(self.lx[i,:]==1)[0])

        return stim.Circuit(circ.circuit)

__all__ = ["BbCode"]
