"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np

from .circuit_construction import get_builder
from ..gf2_util import compute_lz_and_lx
from .base import QldpcCode


class QlpCode(QldpcCode):
    def __init__(self, b1, b2, lift_size):
        '''
        :param b1: First base matrix used to construct the lp code. Each entry is the power of the monomial.
                   e.g. b1 = np.array([[0, 0], [0, 3]]) represents the matrix of monomials [[1, 1], [1, x^3]].
        :param b2: Second base matrix used to construct the lp code. Each entry is the power of the monomial.
        :param lift_size: Size of cyclic matrix to which each monomial entry is lifted.
        '''
        super().__init__()

        self.b1, self.b2 = b1, b2
        self.lift_size = lift_size
        self.m1, self.n1 = b1.shape
        self.m2, self.n2 = b2.shape

        b1T = (self.lift_size - b1).T % self.lift_size
        b2T = (self.lift_size - b2).T % self.lift_size
        b1_placeholder = np.ones(b1.shape, dtype=int)
        b2_placeholder = np.ones(b2.shape, dtype=int)

        hz_base = np.concatenate((np.kron(b2, np.eye(self.n1, dtype=int)),
                                  np.kron(np.eye(self.m2, dtype=int), b1T)), axis=1)
        hx_base = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1),
                                  np.kron(b2T, np.eye(self.m1, dtype=int))), axis=1)
        hz_base_placeholder = np.concatenate((np.kron(b2_placeholder, np.eye(self.n1, dtype=int)),
                                              np.kron(np.eye(self.m2, dtype=int), b1_placeholder.T)), axis=1)
        hx_base_placeholder = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1_placeholder),
                                              np.kron(b2_placeholder.T, np.eye(self.m1, dtype=int))), axis=1)

        self.hz = self.lift(self.lift_size, hz_base, hz_base_placeholder)
        self.hx = self.lift(self.lift_size, hx_base, hx_base_placeholder)
        self.lz, self.lx = compute_lz_and_lx(self.hz, self.hx)

    def build_circuit(self, strategy="cardinal", **opts):
        if strategy != "cardinal":
            return super().build_circuit(strategy=strategy, **opts)
        return self._build_cardinal_graph(**opts)

    def _build_cardinal_graph(self, seed=1):
        get_builder("cardinal").build_graph(self)
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    data_qubits += [node]
                    self.graph.add_node(node, pos=(i, j))
                    self.node_colors += ['blue']

        start = self.n1 * self.lift_size
        for i in range(self.m1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    xcheck_qubits += [node]
                    self.graph.add_node(node, pos=(i + self.n1, j))
                    self.node_colors += ['purple']

        start = self.n2 * (self.n1 + self.m1) * self.lift_size
        for i in range(self.n1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    zcheck_qubits += [node]
                    self.graph.add_node(node, pos=(i, j + self.n2))
                    self.node_colors += ['green']

        start = (self.n2 * (self.n1 + self.m1) + self.n1) * self.lift_size
        for i in range(self.m1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    data_qubits += [node]
                    self.graph.add_node(node, pos=(i + self.n1, j + self.n2))
                    self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))

        hedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)
        vedge_bool_list = self.get_classical_edge_bools(np.ones(self.b2.shape, dtype=int), seed)

        edge_no = 0
        for i in range(self.m1):
            for j in range(self.n1):
                shift = self.b1[i, j]
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n2 + self.m2):
                        if (k < self.n2) ^ edge_bool:
                            direction_ind = self.direction_inds['E']
                        else:
                            direction_ind = self.direction_inds['W']

                        control = (k * (self.n1 + self.m1) + self.n1 + i) * self.lift_size + (l + shift) % self.lift_size
                        target = (k * (self.n1 + self.m1) + j) * self.lift_size + l
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        for i in range(self.m2):
            for j in range(self.n2):
                shift = self.b2[i, j]
                edge_bool = vedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n1 + self.m1):
                        if (k < self.n1) ^ edge_bool:
                            direction_ind = self.direction_inds['N']
                        else:
                            direction_ind = self.direction_inds['S']

                        control = (k + j * (self.n1 + self.m1)) * self.lift_size + l
                        target = (k + (i + self.n2) * (self.n1 + self.m1)) * self.lift_size + (l + shift) % self.lift_size
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return


class QlpPolyCode(QldpcCode):
    def __init__(self, b1, b2, lift_size):
        '''
        :param b1: First base matrix used to construct the lp code. Each entry is the list of powers of the polynomial terms.
                   e.g. b1 = [[[0], [0,1], []], [[], [0], [0,1]]] represents the matrix of monomials [[1, 1+x, 0], [0, 1, 1+x]].
        :param b2: Second base matrix used to construct the lp code. Each entry is the list of powers of the polynomial terms.
        :param lift_size: Size of cyclic matrix to which each polynomial term is lifted.
        '''
        super().__init__()

        self.b1, self.b2 = b1, b2
        self.lift_size = lift_size

        self.m1, self.n1 = len(b1), len(b1[0])
        self.m2, self.n2 = len(b2), len(b2[0])

        # Base matrices where each entry ENCODEs the powers of polynomial terms in base of lift_size
        b1_enc = np.zeros((self.m1, self.n1), dtype=int)
        b1T_enc = np.zeros((self.n1, self.m1), dtype=int)
        b2_enc = np.zeros((self.m2, self.n2), dtype=int)
        b2T_enc = np.zeros((self.n2, self.m2), dtype=int)
        self.b1_placeholder = np.zeros((self.m1, self.n1), dtype=int)
        self.b2_placeholder = np.zeros((self.m2, self.n2), dtype=int)

        for i in range(self.m1):
            for j in range(self.n1):
                if self.b1[i][j] == []:
                    continue
                self.b1_placeholder[i, j] = 1

                bij, bTij = 0, 0
                for k in range(len(self.b1[i][j])):
                    bij += self.lift_size ** k * self.b1[i][j][k]
                    bTij += self.lift_size ** k * ((self.lift_size - self.b1[i][j][k]) % self.lift_size)
                b1_enc[i, j] = bij
                b1T_enc[j, i] = bTij

        for i in range(self.m2):
            for j in range(self.n2):
                if self.b2[i][j] == []:
                    continue
                self.b2_placeholder[i, j] = 1

                bij, bTij = 0, 0
                for k in range(len(self.b2[i][j])):
                    bij += self.lift_size ** k * self.b2[i][j][k]
                    bTij += self.lift_size ** k * ((self.lift_size - self.b2[i][j][k]) % self.lift_size)
                b2_enc[i, j] = bij
                b2T_enc[j, i] = bTij

        hz_base_enc = np.concatenate((np.kron(b2_enc, np.eye(self.n1, dtype=int)),
                                      np.kron(np.eye(self.m2, dtype=int), b1T_enc)), axis=1)
        hx_base_enc = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1_enc),
                                      np.kron(b2T_enc, np.eye(self.m1, dtype=int))), axis=1)
        hz_base_placeholder = np.concatenate((np.kron(self.b2_placeholder, np.eye(self.n1, dtype=int)),
                                              np.kron(np.eye(self.m2, dtype=int), self.b1_placeholder.T)), axis=1)
        hx_base_placeholder = np.concatenate((np.kron(np.eye(self.n2, dtype=int), self.b1_placeholder),
                                              np.kron(self.b2_placeholder.T, np.eye(self.m1, dtype=int))), axis=1)

        self.hz = self.lift_enc(self.lift_size, hz_base_enc, hz_base_placeholder)
        self.hx = self.lift_enc(self.lift_size, hx_base_enc, hx_base_placeholder)
        self.lz, self.lx = compute_lz_and_lx(self.hz, self.hx)

    def build_circuit(self, strategy="cardinal", **opts):
        if strategy != "cardinal":
            return super().build_circuit(strategy=strategy, **opts)
        return self._build_cardinal_graph(**opts)

    def _build_cardinal_graph(self, seed=1):
        get_builder("cardinal").build_graph(self)
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    data_qubits += [node]
                    self.graph.add_node(node, pos=(i, j))
                    self.node_colors += ['blue']

        start = self.n1 * self.lift_size
        for i in range(self.m1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    xcheck_qubits += [node]
                    self.graph.add_node(node, pos=(i + self.n1, j))
                    self.node_colors += ['purple']

        start = self.n2 * (self.n1 + self.m1) * self.lift_size
        for i in range(self.n1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    zcheck_qubits += [node]
                    self.graph.add_node(node, pos=(i, j + self.n2))
                    self.node_colors += ['green']

        start = (self.n2 * (self.n1 + self.m1) + self.n1) * self.lift_size
        for i in range(self.m1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l
                    data_qubits += [node]
                    self.graph.add_node(node, pos=(i + self.n1, j + self.n2))
                    self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))

        hedge_bool_list = self.get_classical_edge_bools(self.b1_placeholder, seed)
        vedge_bool_list = self.get_classical_edge_bools(self.b2_placeholder, seed)

        edge_no = 0
        for i in range(self.m1):
            for j in range(self.n1):
                if self.b1_placeholder[i, j] == 0:
                    continue
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n2 + self.m2):
                        if (k < self.n2) ^ edge_bool:
                            direction_ind = self.direction_inds['E']
                        else:
                            direction_ind = self.direction_inds['W']

                        for shift in self.b1[i][j]:
                            control = (k * (self.n1 + self.m1) + self.n1 + i) * self.lift_size + (l + shift) % self.lift_size
                            target = (k * (self.n1 + self.m1) + j) * self.lift_size + l
                            self.add_edge(edge_no, direction_ind, control, target)
                            edge_no += 1

        for i in range(self.m2):
            for j in range(self.n2):
                if self.b2_placeholder[i, j] == 0:
                    continue
                edge_bool = vedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n1 + self.m1):
                        if (k < self.n1) ^ edge_bool:
                            direction_ind = self.direction_inds['N']
                        else:
                            direction_ind = self.direction_inds['S']

                        for shift in self.b2[i][j]:
                            control = (k + j * (self.n1 + self.m1)) * self.lift_size + l
                            target = (k + (i + self.n2) * (self.n1 + self.m1)) * self.lift_size + (l + shift) % self.lift_size
                            self.add_edge(edge_no, direction_ind, control, target)
                            edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return


__all__ = ["QlpCode", "QlpPolyCode"]
