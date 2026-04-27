import networkx as nx
import numpy as np

from .base import CircuitBuilder, EdgeLayering


class CustomBuilder(CircuitBuilder):
    name = "custom"

    def __init__(self, code):
        if code is None:
            raise ValueError("CustomBuilder requires a code instance.")
        self.code = code
        self.code.set_draw_graph(self.draw_graph)

    def _require_custom_strategy_capabilities(self):
        missing = []
        if not hasattr(self.code, "_ensure_custom_qubit_indexing"):
            missing.append("_ensure_custom_qubit_indexing")
        if not hasattr(self.code, "get_custom_schedule_edges"):
            missing.append("get_custom_schedule_edges")
        if getattr(self.code, "hz", None) is None or getattr(self.code, "hx", None) is None:
            missing.append("hz/hx parity checks")
        if missing:
            missing_list = ", ".join(missing)
            raise TypeError(
                "CustomBuilder requires a code that implements the custom-circuit interface: "
                f"{missing_list}."
            )

    def _ensure_custom_qubit_indexing(self):
        self._require_custom_strategy_capabilities()
        self.code._ensure_custom_qubit_indexing()

    def build(self, code, **opts):
        self.code = code
        self.code.set_draw_graph(self.draw_graph)
        self._ensure_custom_qubit_indexing()
        self.build_graph(**opts)
        return self.code.graph

    def build_graph(self, **opts):
        code = self.code
        self._ensure_custom_qubit_indexing()
        schedule_edges = code.get_custom_schedule_edges()

        code.graph = nx.Graph()
        code.edges_Z = []
        code.edges_X = []
        code.node_colors = {}
        code.custom_schedule_edges = schedule_edges
        code.custom_colored_edges = {
            round_index: schedule_edges[f"round{round_index + 1}"]
            for round_index in range(len(schedule_edges))
        }
        code.custom_num_colors = len(code.custom_colored_edges)
        code.depth = code.custom_num_colors

        fallback_pos = {}
        for idx, node in enumerate(code.xcheck_qubits):
            node = int(node)
            fallback_pos[node] = (0.0, float(idx))
            code.graph.add_node(node, pos=fallback_pos[node])
            code.node_colors[node] = "purple"

        for idx, node in enumerate(code.data_qubits):
            node = int(node)
            fallback_pos[node] = (1.0, float(idx))
            code.graph.add_node(node, pos=fallback_pos[node])
            code.node_colors[node] = "blue"

        for idx, node in enumerate(code.zcheck_qubits):
            node = int(node)
            fallback_pos[node] = (2.0, float(idx))
            code.graph.add_node(node, pos=fallback_pos[node])
            code.node_colors[node] = "green"

        for z_row, data_col in np.argwhere(code.hz == 1):
            data_node = int(code.data_qubits[int(data_col)])
            z_node = int(code.zcheck_qubits[int(z_row)])
            code.edges_Z.append((data_node, z_node))
            code.graph.add_edge(data_node, z_node, basis="Z")

        for x_row, data_col in np.argwhere(code.hx == 1):
            x_node = int(code.xcheck_qubits[int(x_row)])
            data_node = int(code.data_qubits[int(data_col)])
            code.edges_X.append((x_node, data_node))
            code.graph.add_edge(x_node, data_node, basis="X")

        return code.graph

    def _resolve_positions(self, graph, layout=None, **kwargs):
        code = self.code
        if (
            layout is not None
            and code.data_qubits is not None
            and code.zcheck_qubits is not None
            and code.xcheck_qubits is not None
        ):
            pos = layout.node_positions(
                data_qubits=code.data_qubits,
                zcheck_qubits=code.zcheck_qubits,
                xcheck_qubits=code.xcheck_qubits,
            )
            if all(node in pos for node in graph.nodes()):
                return {node: pos[node] for node in graph.nodes()}
        pos = nx.get_node_attributes(graph, "pos")
        if not pos:
            raise ValueError("CustomBuilder.draw_graph requires node positions or a layout.")
        return {node: pos[node] for node in graph.nodes() if node in pos}

    def _normalize_draw_part(self, part):
        return super()._normalize_draw_part(part)

    def _get_graph_for_draw(self, part):
        return self.build_graph()

    def _get_node_colors(self, graph):
        return [self.code.node_colors[node] for node in graph.nodes()]

    def _get_edge_layering(self, graph):
        layers = []
        for round_index in sorted(self.code.custom_colored_edges):
            layers.append(list(self.code.custom_colored_edges[round_index]))
        return EdgeLayering(layers=layers)
