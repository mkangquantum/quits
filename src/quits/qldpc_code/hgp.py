"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np

from ..noise import ErrorModel
from .circuit_construction import get_builder
from .circuit_construction.circuit_build_options import CircuitBuildOptions
from ..gf2_util import gf2_nullspace_basis, gf2_coset_reps_rowspace
from .base import QldpcCode


class HgpCode(QldpcCode):
    supported_strategies = {"cardinal", "zxcoloration"}

    def __init__(self, h1, h2):
        '''
        :param h1: Parity check matrix of the first classical code used to construct the hgp code
        :param h2: Parity check matrix of the second classical code used to construct the hgp code
        '''
        # Reference: Tillich & Zemor, arXiv:0903.0566 (hypergraph product codes).
        super().__init__()

        self.h1, self.h2 = h1, h2
        self.r1, self.n1 = h1.shape
        self.r2, self.n2 = h2.shape

        self.hz = np.concatenate((np.kron(h2, np.eye(self.n1, dtype=int)),
                                  np.kron(np.eye(self.r2, dtype=int), h1.T)), axis=1)
        self.hx = np.concatenate((np.kron(np.eye(self.n2, dtype=int), h1),
                                  np.kron(h2.T, np.eye(self.r1, dtype=int))), axis=1)

        self.l1 = gf2_nullspace_basis(self.h1)
        self.l2 = gf2_nullspace_basis(self.h2)
        self.k1, self.k2 = self.l1.shape[0], self.l2.shape[0]
        self.l1t = gf2_nullspace_basis(self.h1.T)
        self.l2t = gf2_nullspace_basis(self.h2.T)
        self.k1t, self.k2t = self.l1t.shape[0], self.l2t.shape[0]

        self.lz, self.lx = self.get_canonical_logicals()

    def get_canonical_logicals(self):
        """
        Canonical logicals for the HGP convention, including both sectors:
          - VV sector: k1*k2 logicals
          - CC sector: k1^T*k2^T logicals
        See arXiv:2204.10812.

        Returns:
          lz, lx: shape (k1*k2 + k1^T*k2^T, num_data_qubits) as uint8.
        """
        E1 = gf2_coset_reps_rowspace(self.h1)      # (k1, n1)
        E2 = gf2_coset_reps_rowspace(self.h2)      # (k2, n2)
        E1t = gf2_coset_reps_rowspace(self.h1.T)   # (k1^T, r1)
        E2t = gf2_coset_reps_rowspace(self.h2.T)   # (k2^T, r2)

        k_vv = self.k1 * self.k2
        k_cc = self.k1t * self.k2t
        k_total = k_vv + k_cc
        split = self.n1 * self.n2

        lz = np.zeros((k_total, self.hz.shape[1]), dtype=np.uint8)
        lx = np.zeros((k_total, self.hx.shape[1]), dtype=np.uint8)

        cnt = 0
        for i in range(self.k2):
            for j in range(self.k1):
                # VV sector
                # Z: (E2_i \otimes L1_j | 0)
                lz[cnt, :split] = np.kron(E2[i, :], self.l1[j, :]) & 1
                # X: (L2_i \otimes E1_j | 0)
                lx[cnt, :split] = np.kron(self.l2[i, :], E1[j, :]) & 1
                cnt += 1

        for i in range(self.k2t):
            for j in range(self.k1t):
                # CC sector
                # Z: (0 | L2^T_i \otimes E1^T_j)
                lz[cnt, split:] = np.kron(self.l2t[i, :], E1t[j, :]) & 1
                # X: (0 | E2^T_i \otimes L1^T_j)
                lx[cnt, split:] = np.kron(E2t[i, :], self.l1t[j, :]) & 1
                cnt += 1

        return lz, lx

    def build_circuit(
        self,
        strategy="cardinal",
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        **opts,
    ):
        '''
        Build a circuit for this HGP code using the selected construction strategy.

        :param strategy: Circuit-construction strategy name (e.g., "cardinal").
        :param error_model: ErrorModel specifying idle/single-/two-qubit/SPAM noise.
        :param num_rounds: Number of noisy syndrome-extraction rounds after the zeroth round.
        :param basis: Logical storage/measurement basis, either "Z" or "X".
        :param circuit_build_options: CircuitBuildOptions controlling detector and noise toggles.
        :param opts: Additional keyword arguments, e.g., seed for the cardinal strategy.
        :return: Stim circuit.
        '''
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")
        
        if strategy == "cardinal":
            seed = opts.get("seed", 1)
            return self._build_cardinal_circuit(
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
                seed=seed,
            )
        elif strategy == "zxcoloration":
            builder = get_builder("zxcoloration", self)
            return builder.get_coloration_circuit(
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
            )
        else:
            return super().build_circuit(strategy=strategy, **opts)

    def _build_cardinal_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        seed=1,
    ):
        """
        Build a cardinal circuit for this HGP code.

        :param seed: Random seed used by graph-edge orientation/coloring helpers.
        """
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")
        builder = get_builder("cardinal", self)
        builder.build_graph()
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                node = i + j * (self.n1 + self.r1)
                data_qubits += [node]
                self.graph.add_node(node, pos=(i, j))

        start = self.n1
        for i in range(self.r1):
            for j in range(self.n2):
                node = start + i + j * (self.n1 + self.r1)
                xcheck_qubits += [node]
                self.graph.add_node(node, pos=(i + self.n1, j))

        start = self.n2 * (self.n1 + self.r1)
        for i in range(self.n1):
            for j in range(self.r2):
                node = start + i + j * (self.n1 + self.r1)
                zcheck_qubits += [node]
                self.graph.add_node(node, pos=(i, j + self.n2))

        start = self.n2 * (self.n1 + self.r1) + self.n1
        for i in range(self.r1):
            for j in range(self.r2):
                node = start + i + j * (self.n1 + self.r1)
                data_qubits += [node]
                self.graph.add_node(node, pos=(i + self.n1, j + self.n2))

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))                

        hedge_bool_list = self.get_classical_edge_bools(self.h1, seed)
        vedge_bool_list = self.get_classical_edge_bools(self.h2, seed)

        for classical_edge in np.argwhere(self.h1 == 1):
            c0, c1 = classical_edge
            edge_bool = hedge_bool_list[(c0, c1)]
            for k in range(self.n2 + self.r2):
                control, target = (k * (self.n1 + self.r1) + c0 + self.n1, k * (self.n1 + self.r1) + c1)
                if (k < self.n2) ^ edge_bool:
                    direction = 'E'
                else:
                    direction = 'W'
                self.add_edge(direction, control, target)

        for classical_edge in np.argwhere(self.h2 == 1):
            c0, c1 = classical_edge
            edge_bool = vedge_bool_list[(c0, c1)]
            for k in range(self.n1 + self.r1):
                control, target = (k + c1 * (self.n1 + self.r1), k + (c0 + self.n2) * (self.n1 + self.r1))
                if (k < self.n1) ^ edge_bool:
                    direction = 'N'
                else:
                    direction = 'S'
                self.add_edge(direction, control, target)

        # Color the edges of self.graph
        self.color_edges()
        return builder.get_cardinal_circuit(
            error_model=error_model,
            num_rounds=num_rounds,
            basis=basis,
            circuit_build_options=circuit_build_options,
        )


__all__ = ["HgpCode"]
