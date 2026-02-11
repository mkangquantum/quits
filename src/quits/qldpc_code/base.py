"""
@author: Mingyu Kang, Yingjia Lin
"""

import warnings

import numpy as np
from scipy.linalg import circulant
from ..gf2_util import verify_css_logicals
from .circuit_construction import get_builder


class QldpcCode:
    supported_strategies = {"zxcoloration"}

    def __init__(self):

        self.hz, self.hx = None, None
        self.lz, self.lx = None, None

        self.data_qubits, self.zcheck_qubits, self.xcheck_qubits = None, None, None
        self.check_qubits, self.all_qubits = None, None

    def verify_css_logicals(self):
        return verify_css_logicals(self.hz, self.hx, self.lz, self.lx)

    def get_circulant_mat(self, size, power):
        return circulant(np.eye(size, dtype=int)[:, power])

    def lift(self, lift_size, h_base, h_base_placeholder):
        '''
        :param lift_size: Size of cyclic matrix to which each monomial entry is lifted.
        :param h_base: Base matrix where each entry is the power of the monomial.
        :param h_base_placeholder: Placeholder matrix where each non-zero entry of the base matrix is replaced by 1.
        :return: Lifted matrix.
        '''
        h = np.zeros((h_base.shape[0] * lift_size, h_base.shape[1] * lift_size), dtype=int)
        for i in range(h_base.shape[0]):
            for j in range(h_base.shape[1]):
                if h_base_placeholder[i, j] != 0:
                    h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] = self.get_circulant_mat(
                        lift_size, h_base[i, j]
                    )
        return h

    def lift_enc(self, lift_size, h_base_enc, h_base_placeholder):
        '''
        :param lift_size: Size of cyclic matrix to which each polynomial term is lifted.
        :param h_base: Base matrix where each entry ENCODEs the powers of polynomial terms in base of lift_size.
        :param h_base_placeholder: Placeholder matrix where each non-zero entry of the base matrix is replaced by 1.
        :return: Lifted matrix.
        '''
        h = np.zeros((h_base_enc.shape[0] * lift_size, h_base_enc.shape[1] * lift_size), dtype=int)
        for i in range(h_base_enc.shape[0]):
            for j in range(h_base_enc.shape[1]):
                if h_base_placeholder[i, j] != 0:
                    hij_enc = h_base_enc[i, j]
                    if hij_enc == 0:
                        h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] = self.get_circulant_mat(
                            lift_size, 0
                        )
                    else:
                        while hij_enc > 0:
                            power = hij_enc % lift_size
                            h[i * lift_size:(i + 1) * lift_size, j * lift_size:(j + 1) * lift_size] += self.get_circulant_mat(
                                lift_size, power
                            )
                            hij_enc = hij_enc // lift_size
        return h

    # Draw the Tanner graph of the code.
    def draw_graph(self, draw_edges=True):
        builder = get_builder("cardinal", self)
        return builder.draw_graph(draw_edges=draw_edges)

    def build_circuit(self, strategy="zxcoloration", **opts):
        if strategy == "cardinal" and strategy not in self.supported_strategies:
            supported = ", ".join(sorted(self.supported_strategies))
            msg = (
                f"Error: strategy='cardinal' is not supported for {type(self).__name__}. "
                f"Supported strategies: {supported}."
            )
            print(msg)
            raise NotImplementedError(msg)
        builder = get_builder(strategy)
        return builder.build(self, **opts)

    def build_graph(self, **opts):
        warnings.warn(
            "QldpcCode.build_graph is deprecated; use build_circuit(strategy='zxcoloration', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.build_circuit(strategy="cardinal", **opts)

    # Helper function for assigning bool to each edge of the classical code's parity check matrix
    def get_classical_edge_bools(self, h, seed):
        builder = get_builder("cardinal", self)
        return builder.get_classical_edge_bools(h, seed)

    # Helper function for adding edges
    def add_edge(self, direction, control, target):
        builder = get_builder("cardinal", self)
        return builder.add_edge(direction, control, target)

    def color_edges(self):
        builder = get_builder("cardinal", self)
        return builder.color_edges()


__all__ = ["QldpcCode"]
