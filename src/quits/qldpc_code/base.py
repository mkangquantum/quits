"""
@author: Mingyu Kang, Yingjia Lin
"""

import warnings

import numpy as np
from ..gf2_util import compute_lz_and_lx, verify_css_logicals
from .circuit_construction import get_builder
from .qldpc_util import get_circulant_mat as _get_circulant_mat
from .qldpc_util import lift as _lift
from .qldpc_util import lift_enc as _lift_enc


class QldpcCode:
    supported_strategies = {"zxcoloration"}

    def __init__(self):

        self.hz, self.hx = None, None
        self.lz, self.lx = None, None

        self.data_qubits, self.zcheck_qubits, self.xcheck_qubits = None, None, None
        self.check_qubits, self.all_qubits = None, None

    @classmethod
    def from_parity_checks(cls, hz, hx, *, compute_logicals=True):
        """Construct a QldpcCode directly from parity-check matrices."""
        code = cls()
        code.set_parity_checks(hz, hx, compute_logicals=compute_logicals)
        return code

    def set_parity_checks(self, hz, hx, *, compute_logicals=True):
        """Set hz/hx and optionally derive CSS logical operators."""
        hz = (np.asarray(hz) & 1).astype(np.uint8, copy=False)
        hx = (np.asarray(hx) & 1).astype(np.uint8, copy=False)
        if hz.ndim != 2 or hx.ndim != 2:
            raise ValueError("hz and hx must be 2D arrays")
        if hz.shape[1] != hx.shape[1]:
            raise ValueError("hz and hx must have the same number of columns")
        self.hz, self.hx = hz, hx
        if compute_logicals:
            self.lz, self.lx = compute_lz_and_lx(hz, hx)
        else:
            self.lz, self.lx = None, None
        return self

    def verify_css_logicals(self):
        return verify_css_logicals(self.hz, self.hx, self.lz, self.lx)

    def get_circulant_mat(self, size, power):
        warnings.warn(
            "QldpcCode.get_circulant_mat is deprecated; use quits.qldpc_code.qldpc_util.get_circulant_mat instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _get_circulant_mat(size, power)

    def lift(self, lift_size, h_base, h_base_placeholder):
        warnings.warn(
            "QldpcCode.lift is deprecated; use quits.qldpc_code.qldpc_util.lift instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _lift(lift_size, h_base, h_base_placeholder)

    def lift_enc(self, lift_size, h_base_enc, h_base_placeholder):
        warnings.warn(
            "QldpcCode.lift_enc is deprecated; use quits.qldpc_code.qldpc_util.lift_enc instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _lift_enc(lift_size, h_base_enc, h_base_placeholder)

    # Draw the Tanner graph of the code.
    def draw_graph(
        self,
        part="node",
        draw_edges=True,
        x_scale=3.0,
        y_scale=3.0,
        node_size=100,
        font_size=8,
        figsize=None,
    ):
        builder = get_builder("cardinal", self)
        return builder.draw_graph(
            part=part,
            draw_edges=draw_edges,
            x_scale=x_scale,
            y_scale=y_scale,
            node_size=node_size,
            font_size=font_size,
            figsize=figsize,
        )

    def build_circuit(self, strategy="zxcoloration", **opts):
        if strategy == "zxcoloration":
            builder = get_builder("zxcoloration", self)
            return builder.get_coloration_circuit(
                error_model=opts.get("error_model"),
                num_rounds=opts.get("num_rounds", 0),
                basis=opts.get("basis", "Z"),
                circuit_build_options=opts.get("circuit_build_options"),
            )
        if strategy in ("cardinal", "cardinalNSmerge", "custom") and strategy not in self.supported_strategies:
            supported = ", ".join(sorted(self.supported_strategies))
            msg = (
                f"Error: strategy='{strategy}' is not supported for {type(self).__name__}. "
                f"Supported strategies: {supported}."
            )
            print(msg)
            raise NotImplementedError(msg)
        builder = get_builder(strategy, self)
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
