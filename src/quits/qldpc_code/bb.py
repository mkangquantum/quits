"""
Bivariate bicycle (BB) code construction.
"""

import numpy as np

from ..circuit import Circuit
from ..gf2_util import compute_lz_and_lx
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

    def build_circuit(self, strategy="freeform", **opts):
        if strategy != "freeform":
            return super().build_circuit(strategy=strategy, **opts)
        return self._build_freeform_circuit(**opts)

    def _build_freeform_circuit(
        self,
        A_list=None,
        B_list=None,
        p=0.0,
        num_repeat=1,
        z_basis=True,
        use_both=False,
        HZH=False,
    ):
        n = int(self.hx.shape[1])
        if n % 2 != 0:
            raise ValueError("Number of data qubits must be even.")
        if num_repeat < 1:
            raise ValueError("num_repeat must be at least 1.")

        half = n // 2

        if A_list is None or B_list is None:
            S_l = self.get_circulant_mat(self.l, -1)
            S_m = self.get_circulant_mat(self.m, -1)
            x = np.kron(S_l, np.eye(self.m, dtype=int))
            y = np.kron(np.eye(self.l, dtype=int), S_m)

        if A_list is None:
            A_list = [
                np.linalg.matrix_power(x, power) for power in self.A_x_pows
            ] + [
                np.linalg.matrix_power(y, power) for power in self.A_y_pows
            ]
        if B_list is None:
            B_list = [
                np.linalg.matrix_power(y, power) for power in self.B_y_pows
            ] + [
                np.linalg.matrix_power(x, power) for power in self.B_x_pows
            ]

        if len(A_list) != 3 or len(B_list) != 3:
            raise ValueError("A_list and B_list must each contain exactly 3 terms.")

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

        for name, idxs in [
            ("A1", A1),
            ("A2", A2),
            ("A3", A3),
            ("B1", B1),
            ("B2", B2),
            ("B3", B3),
            ("A1_T", A1_T),
            ("A2_T", A2_T),
            ("A3_T", A3_T),
            ("B1_T", B1_T),
            ("B2_T", B2_T),
            ("B3_T", B3_T),
        ]:
            if idxs.size != half:
                raise ValueError(f"{name} must have exactly one nonzero per row.")

        x_check_offset = 0
        l_data_offset = half
        r_data_offset = n
        z_check_offset = n + half

        x_checks = np.arange(x_check_offset, x_check_offset + half, dtype=int)
        z_checks = np.arange(z_check_offset, z_check_offset + half, dtype=int)
        data_qubits = np.arange(l_data_offset, l_data_offset + n, dtype=int)
        all_qubits = np.arange(2 * n, dtype=int)

        circ = Circuit(all_qubits)
        circ.set_error_rates(p, p, p, p)

        circ.add_reset(x_checks)
        circ.add_reset(z_checks)
        circ.add_reset(data_qubits, basis="Z" if z_basis else "X")
        circ.add_tick()

        def make_edges(control_offset, target_offset, mapping):
            return [(control_offset + int(mapping[i]), target_offset + i) for i in range(half)]

        edges_round1 = make_edges(r_data_offset, z_check_offset, A1_T)
        edges_round2 = make_edges(x_check_offset, l_data_offset, A2) + make_edges(
            r_data_offset, z_check_offset, A3_T
        )
        edges_round3 = make_edges(x_check_offset, r_data_offset, B2) + make_edges(
            l_data_offset, z_check_offset, B1_T
        )
        edges_round4 = make_edges(x_check_offset, r_data_offset, B1) + make_edges(
            l_data_offset, z_check_offset, B2_T
        )
        edges_round5 = make_edges(x_check_offset, r_data_offset, B3) + make_edges(
            l_data_offset, z_check_offset, B3_T
        )
        edges_round6 = make_edges(x_check_offset, l_data_offset, A1) + make_edges(
            r_data_offset, z_check_offset, A2_T
        )
        edges_round7 = make_edges(x_check_offset, l_data_offset, A3)

        def flatten(edges):
            return [q for edge in edges for q in edge]

        def add_z_detectors(repeat):
            if z_basis:
                if repeat:
                    for i in range(half):
                        circ.add_detector([half - i, n + half - i])
                else:
                    for i in range(half):
                        circ.add_detector([half - i])
            elif use_both and repeat:
                for i in range(half):
                    circ.add_detector([half - i, n + half - i])

        def add_x_detectors(repeat):
            if not z_basis:
                if repeat:
                    for i in range(half):
                        circ.add_detector([half - i, n + half - i])
                else:
                    for i in range(half):
                        circ.add_detector([half - i])
            elif use_both and repeat:
                for i in range(half):
                    circ.add_detector([half - i, n + half - i])

        def append_block(repeat=False):
            if (not repeat) or HZH:
                circ.add_hadamard_layer(x_checks)

            circ.add_cnot_layer(flatten(edges_round1))
            circ.add_cnot_layer(flatten(edges_round2))
            circ.add_cnot_layer(flatten(edges_round3))
            circ.add_cnot_layer(flatten(edges_round4))
            circ.add_cnot_layer(flatten(edges_round5))
            circ.add_cnot_layer(flatten(edges_round6))
            circ.add_cnot_layer(flatten(edges_round7))

            circ.add_measure_reset_layer(z_checks)
            add_z_detectors(repeat)

            circ.add_hadamard_layer(x_checks)
            circ.add_measure_reset_layer(x_checks)
            if not HZH:
                circ.add_hadamard_layer(x_checks)
            add_x_detectors(repeat)

        for rep in range(num_repeat):
            append_block(repeat=rep > 0)

        circ.add_measure(data_qubits, basis="Z" if z_basis else "X")

        pcm = self.hz if z_basis else self.hx
        logical_pcm = self.lz if z_basis else self.lx

        for i, s in enumerate(pcm):
            nnz_inds = np.nonzero(s)[0]
            det_inds = [n - int(ind) for ind in nnz_inds]
            if z_basis:
                det_inds.append(2 * n - i)
            else:
                det_inds.append(n + half - i)
            circ.add_detector(det_inds)

        for i, l in enumerate(logical_pcm):
            nnz_inds = np.nonzero(l)[0]
            obs_inds = [n - int(ind) for ind in nnz_inds]
            circ.add_observable(i, obs_inds)

        return circ.circuit

__all__ = ["BbCode"]
