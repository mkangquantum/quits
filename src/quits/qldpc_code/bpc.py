"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np

from .circuit_construction import get_builder
from ..gf2_util import _gf2_inv_square, compute_lz_and_lx
from .base import QldpcCode


class BpcCode(QldpcCode):
    def __init__(self, p1, p2, lift_size, factor, verbose=False, canonical_basis="z"):
        '''
        :param p1: First polynomial used to construct the bp code. Each entry of the list is the power of each polynomial term.
                   e.g. p1 = [0, 1, 5] represents the polynomial 1 + x + x^5
        :param p2: Second polynomial used to construct the bp code. Each entry of the list is the power of each polynomial term.
        :param lift_size: Size of cyclic matrix to which each monomial entry is lifted.
        :param factor: Power of the monomial generator of the cyclic subgroup that is factored out by the balanced product.
                       e.g. if factor == 3, cyclic subgroup <x^3> is factored out.
        '''
        # Reference: R. Tiew & N. P. Breuckmann, arXiv:2411.03302 (balanced product cyclic codes).
        # Note: To match the paper, p2 should use lift_size minus the powers listed there (transpose convention).
        super().__init__()

        self.p1, self.p2 = p1, p2
        self.lift_size = lift_size
        self.factor = factor
        self.verbose = verbose
        self.canonical_basis = canonical_basis

        b1 = np.zeros((self.factor, self.factor), dtype=int)
        b1_placeholder = np.zeros((self.factor, self.factor), dtype=int)
        for power in p1:
            mat, mat_placeholder = self.get_block_mat(power)
            b1 = b1 + mat
            b1_placeholder = b1_placeholder + mat_placeholder
        b1T = (self.lift_size - b1.T) % self.lift_size
        b1T_placeholder = b1_placeholder.T

        self.b1, self.b1T = b1, b1T
        self.b1_placeholder, self.b1T_placeholder = b1_placeholder, b1T_placeholder

        h1 = self.lift(self.lift_size, b1, b1_placeholder)
        h1T = self.lift(self.lift_size, b1T, b1T_placeholder)

        h2 = np.zeros((self.lift_size, self.lift_size), dtype=int)
        for power in p2:
            h2 = h2 + self.get_circulant_mat(self.lift_size, power)
        h2 = np.kron(np.eye(self.factor, dtype=int), h2)
        h2T = h2.T

        self.hz = np.concatenate((h2, h1T), axis=1)
        self.hx = np.concatenate((h1, h2T), axis=1)
        # q = lift_size / factor in the balanced product construction.
        q = self.lift_size // self.factor
        if q % 2 == 1:
            if self.verbose:
                print("BpcCode: using canonical logical codewords (q is odd).")
            self.lz, self.lx = self.get_canonical_logicals(canonical_basis=self.canonical_basis)
        else:
            self.lz, self.lx = compute_lz_and_lx(self.hz, self.hx)

    def get_block_mat(self, power):
        gen_mat = self.get_circulant_mat(self.factor, 1)
        gen_mat[0, -1] = 2

        mat = np.linalg.matrix_power(gen_mat, power)
        mat_placeholder = (mat > 0) * 1

        mat = np.log2(mat + 1e-8).astype(int)
        mat = mat * mat_placeholder * self.factor
        return mat, mat_placeholder

    # WRONG; SHOULD BE FIXED LATER
    def get_canonical_logicals(self, canonical_basis="z"):
        '''
        :return: Logical operators of the code as a list of tuples (logical_z, logical_x)
                 where logical_z and logical_x are numpy arrays of shape (num_logicals, num_data_qubits)
                 The logicals are written in the "canonical form" as described in Eq. 30 of arXiv:2411.03302
        '''

        lz = np.zeros((2 * (self.factor - 1) ** 2, self.hz.shape[1]), dtype=int)
        lx = np.zeros((2 * (self.factor - 1) ** 2, self.hx.shape[1]), dtype=int)

        cnt = 0
        for i in range(self.factor - 1):
            for j in range(self.factor - 1):
                yi_vec = self.get_circulant_mat(self.factor, 0)[:, i]
                xjgx_vec = (self.get_circulant_mat(self.factor, 0) + self.get_circulant_mat(self.factor, 1))[:, j]
                xjgx_vec = np.tile(xjgx_vec, self.lift_size // self.factor)

                prod = np.kron(yi_vec, xjgx_vec)
                lz[cnt, :] = np.concatenate((np.zeros(self.hz.shape[1] - len(prod), dtype=int), prod))
                lx[cnt, :] = np.concatenate((prod, np.zeros(self.hx.shape[1] - len(prod), dtype=int)))

                cnt += 1

        for i in range(self.factor - 1):
            for j in range(self.factor - 1):
                yigy_vec = (self.get_circulant_mat(self.factor, 0) + self.get_circulant_mat(self.factor, 1))[:, i]
                xj_vec = self.get_circulant_mat(self.factor, 0)[:, j]
                xj_vec = np.tile(xj_vec, self.lift_size // self.factor)

                prod = np.kron(yigy_vec, xj_vec)
                lz[cnt, :] = np.concatenate((prod, np.zeros(self.hz.shape[1] - len(prod), dtype=int)))
                lx[cnt, :] = np.concatenate((np.zeros(self.hx.shape[1] - len(prod), dtype=int), prod))

                cnt += 1

        if canonical_basis == "z":
            pairing = (lz @ lx.T) & 1
            inv_pairing = _gf2_inv_square(pairing)
            lx = (inv_pairing.T @ lx) & 1
        elif canonical_basis == "x":
            pairing = (lx @ lz.T) & 1
            inv_pairing = _gf2_inv_square(pairing)
            lz = (inv_pairing @ lz) & 1

        return lz, lx

    def build_circuit(self, strategy="cardinal", **opts):
        if strategy != "cardinal":
            return super().build_circuit(strategy=strategy, **opts)
        return self._build_cardinal_graph(**opts)

    def _build_cardinal_graph(self, seed=1):
        get_builder("cardinal").build_graph(self)
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = i * self.lift_size + l
                data_qubits += [node]
                self.graph.add_node(node, pos=(2 * i, 0))
                self.node_colors += ['blue']

        start = self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                xcheck_qubits += [node]
                self.graph.add_node(node, pos=(2 * i + 1, 0))
                self.node_colors += ['purple']

        start = 2 * self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                zcheck_qubits += [node]
                self.graph.add_node(node, pos=(2 * i, 1))
                self.node_colors += ['green']

        start = 3 * self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                data_qubits += [node]
                self.graph.add_node(node, pos=(2 * i + 1, 1))
                self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))

        hedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)
        vedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)

        # Add edges to the Tanner graph of each direction
        edge_no = 0
        for i in range(self.factor):
            for j in range(self.factor):
                shift = self.b1[i, j]
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(2):  # 0 : bottom, 1 : top
                        if k ^ edge_bool:
                            direction_ind = self.direction_inds['E']
                        else:
                            direction_ind = self.direction_inds['W']

                        control = (2 * k + 1) * self.factor * self.lift_size + i * self.lift_size + (l + shift) % self.lift_size
                        target = 2 * k * self.factor * self.lift_size + j * self.lift_size + l
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        def shuffle(node_no, qubit_no):
            m, r = qubit_no // self.factor, qubit_no % self.factor
            return r, self.lift_size // self.factor * node_no + m

        for i in range(self.factor):
            for j in range(len(self.p2)):
                shift = self.p2[j]

                for l in range(self.lift_size):
                    for k in range(2):  # 0 : left, 1 : right
                        i_shuffled, _ = shuffle(i, l)
                        j_shuffled, _ = shuffle(i, (l + shift) % self.lift_size)
                        edge_bool = vedge_bool_list[(i_shuffled, j_shuffled)]
                        if k ^ edge_bool:
                            direction_ind = self.direction_inds['N']
                        else:
                            direction_ind = self.direction_inds['S']

                        control = k * self.factor * self.lift_size + i * self.lift_size + l
                        target = (2 + k) * self.factor * self.lift_size + i * self.lift_size + (l + shift) % self.lift_size
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return


__all__ = ["BpcCode"]
