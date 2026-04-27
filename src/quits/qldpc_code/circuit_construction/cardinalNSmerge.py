import random

import networkx as nx
import numpy as np
import stim

from ...circuit import Circuit
from ...noise import ErrorModel
from .base import CircuitBuilder, EdgeLayering
from .circuit_build_options import CircuitBuildOptions
from .edge_coloration import edge_coloration


class CardinalNSMergeBuilder(CircuitBuilder):
    name = "cardinalNSmerge"

    def __init__(self, code=None):
        self.code = code
        if code is not None:
            code.set_draw_graph(self.draw_graph)

    def build(self, code, **opts):
        self.code = code
        self.code.set_draw_graph(self.draw_graph)
        self.build_graph(**opts)
        return self.code.graph

    def draw_graph(
        self,
        layout=None,
        part="all",
        draw_edges=True,
        x_scale=3.0,
        y_scale=3.0,
        center_checks=True,
        curved_edges=False,
        node_size=100,
        font_size=8,
        figsize=None,
    ):
        if isinstance(part, str):
            normalized = part.strip().lower()
            if normalized == "color":
                part = "all"
            elif normalized == "direction":
                return self._draw_direction_graph(
                    layout=layout,
                    draw_edges=draw_edges,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    center_checks=center_checks,
                    node_size=node_size,
                    font_size=font_size,
                    figsize=figsize,
                )

        return super().draw_graph(
            layout=layout,
            part=part,
            draw_edges=draw_edges,
            x_scale=x_scale,
            y_scale=y_scale,
            center_checks=center_checks,
            curved_edges=curved_edges,
            node_size=node_size,
            font_size=font_size,
            figsize=figsize,
        )

    def _draw_direction_graph(
        self,
        layout=None,
        draw_edges=True,
        x_scale=3.0,
        y_scale=3.0,
        center_checks=True,
        node_size=100,
        font_size=8,
        figsize=None,
    ):
        graph = self._get_graph_for_draw("direction")
        pos = self._resolve_positions(graph, layout=layout, center_checks=center_checks)
        if x_scale != 1.0 or y_scale != 1.0:
            pos = {key: (value[0] * x_scale, value[1] * y_scale) for key, value in pos.items()}

        if figsize is not None:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=figsize)
            except ImportError:
                pass

        nodes = list(graph.nodes())
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=self._get_node_colors(graph),
            node_size=node_size,
        )

        if draw_edges:
            direction_color_map = {
                "E": "tab:green",
                "N": "tab:blue",
                "S": "tab:orange",
                "W": "tab:red",
            }
            direction_by_edge = {}
            for direction in ("E", "N", "S", "W"):
                for u, v in getattr(self.code, f"edges_{direction}", []):
                    direction_by_edge[frozenset((u, v))] = direction

            edges = list(graph.edges())
            edge_colors = [
                direction_color_map.get(direction_by_edge.get(frozenset((u, v))), "tab:gray")
                for u, v in edges
            ]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors)

        nx.draw_networkx_labels(graph, pos, font_size=font_size, font_color="white")

        try:
            import matplotlib.pyplot as plt

            plt.gca().set_axis_off()
        except ImportError:
            pass

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
        return nx.get_node_attributes(graph, "pos")

    def _get_node_colors(self, graph):
        code = self.code
        nodes = list(graph.nodes())
        data_set = set(int(q) for q in code.data_qubits)
        zcheck_set = set(int(q) for q in code.zcheck_qubits)
        xcheck_set = set(int(q) for q in code.xcheck_qubits)

        def _node_color(node):
            if node in data_set:
                return "blue"
            if node in zcheck_set:
                return "green"
            if node in xcheck_set:
                return "purple"
            return "gray"

        return [_node_color(node) for node in nodes]

    def _get_edge_layering(self, graph):
        code = self.code
        colored_edges_by_direction = {
            "E": getattr(code, "colored_edges_E", {}),
            "NS": getattr(code, "colored_edges_NS", {}),
            "W": getattr(code, "colored_edges_W", {}),
        }
        layers = []
        for direction in ("E", "NS", "W"):
            for color in sorted(colored_edges_by_direction[direction]):
                layers.append(list(colored_edges_by_direction[direction][color]))
        return EdgeLayering(layers=layers)

    def build_graph(self, **opts):
        code = self.code
        code.graph = nx.Graph()

        code.edges_E = []
        code.edges_N = []
        code.edges_S = []
        code.edges_W = []

        code.colored_edges_E = {}
        code.colored_edges_NS = {}
        code.colored_edges_W = {}
        code.num_colors = {"E": 0, "NS": 0, "W": 0}
        return code.graph

    def get_classical_edge_bools(self, h, seed):
        c0_scores = {}
        c1_scores = {}
        edge_signs = {}
        random.seed(seed)

        for edge in np.argwhere(h == 1):
            c0, c1 = edge
            c0_score = c0_scores.get(c0, 0)
            c1_score = c1_scores.get(c1, 0)

            p = random.random()
            tf = c0_score + c1_score > 0 or (c0_score + c1_score == 0 and p >= 0.5)
            sign = int(tf) * 2 - 1
            edge_signs[(c0, c1)] = tf
            c0_scores[c0] = c0_scores.get(c0, 0) - sign
            c1_scores[c1] = c1_scores.get(c1, 0) - sign

        return edge_signs

    def add_edge(self, direction, control, target):
        code = self.code
        edge = (control, target)
        if direction == "E":
            code.edges_E.append(edge)
        elif direction == "N":
            code.edges_N.append(edge)
        elif direction == "S":
            code.edges_S.append(edge)
        elif direction == "W":
            code.edges_W.append(edge)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        code.graph.add_edge(control, target)

    def color_edges(self):
        code = self.code

        def _build_direction_graph(edges):
            graph = nx.Graph()
            graph.add_nodes_from([int(q) for q in code.data_qubits], bipartite=0)
            graph.add_nodes_from([int(q) for q in code.check_qubits], bipartite=1)
            for control, target in edges:
                graph.add_edge(control, target, orientation=(control, target))
            return graph

        code.colored_edges_E = edge_coloration(_build_direction_graph(code.edges_E))
        code.colored_edges_NS = edge_coloration(_build_direction_graph(code.edges_N + code.edges_S))
        code.colored_edges_W = edge_coloration(_build_direction_graph(code.edges_W))

        code.num_colors = {
            "E": len(code.colored_edges_E),
            "NS": len(code.colored_edges_NS),
            "W": len(code.colored_edges_W),
        }
        code.depth = sum(code.num_colors.values())

    def get_cardinal_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
    ):
        code = self.code
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")

        basis = basis.upper()
        if basis not in ("Z", "X"):
            raise ValueError("basis must be 'Z' or 'X'")
        get_z_detectors = basis == "Z" or circuit_build_options.get_all_detectors
        get_x_detectors = basis == "X" or circuit_build_options.get_all_detectors
        directions = ["E", "NS", "W"]
        circ = Circuit(code.all_qubits)

        def _add_stabilizer_round():
            def _flatten_edge_pairs(edge_pairs):
                return [q for control, target in edge_pairs for q in (control, target)]

            colored_edges_by_direction = {
                "E": code.colored_edges_E,
                "NS": code.colored_edges_NS,
                "W": code.colored_edges_W,
            }
            circ.add_hadamard_layer(code.xcheck_qubits)
            for direction in directions:
                for color in range(code.num_colors[direction]):
                    edges = colored_edges_by_direction[direction][color]
                    circ.add_cnot_layer(_flatten_edge_pairs(edges))
            circ.add_hadamard_layer(code.xcheck_qubits)
            circ.add_measure_reset_layer(code.check_qubits)

        if circuit_build_options.noisy_zeroth_round:
            circ.set_error_model(error_model)
        else:
            circ.set_error_model(ErrorModel.zero())

        circ.add_reset(code.data_qubits, basis)
        circ.add_reset(code.check_qubits)
        circ.add_tick()

        _add_stabilizer_round()

        if basis == "Z":
            for i in range(len(code.zcheck_qubits), 0, -1):
                circ.add_detector([len(code.xcheck_qubits) + i])
        else:
            for i in range(len(code.xcheck_qubits), 0, -1):
                circ.add_detector([i])

        circ.set_error_model(error_model)

        if num_rounds > 0:
            circ.start_loop(num_rounds)

            _add_stabilizer_round()

            if get_z_detectors:
                for i in range(len(code.zcheck_qubits), 0, -1):
                    ind = len(code.xcheck_qubits) + i
                    circ.add_detector([ind, ind + len(code.check_qubits)])
            if get_x_detectors:
                for i in range(len(code.xcheck_qubits), 0, -1):
                    circ.add_detector([i, i + len(code.check_qubits)])

            circ.end_loop()

        if not circuit_build_options.noisy_final_meas:
            circ.set_error_model(ErrorModel.zero())

        circ.add_measure(code.data_qubits, basis)

        if basis == "Z":
            for i in range(len(code.zcheck_qubits), 0, -1):
                inds = np.array([len(code.data_qubits) + len(code.xcheck_qubits) + i])
                inds = np.concatenate(
                    (
                        inds,
                        len(code.data_qubits)
                        - np.where(code.hz[len(code.zcheck_qubits) - i, :] == 1)[0],
                    )
                )
                circ.add_detector(inds)

            for i in range(len(code.lz)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lz[i, :] == 1)[0])
        else:
            for i in range(len(code.xcheck_qubits), 0, -1):
                inds = np.array([len(code.data_qubits) + i])
                inds = np.concatenate(
                    (
                        inds,
                        len(code.data_qubits)
                        - np.where(code.hx[len(code.xcheck_qubits) - i, :] == 1)[0],
                    )
                )
                circ.add_detector(inds)

            for i in range(len(code.lx)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lx[i, :] == 1)[0])

        return stim.Circuit(circ.circuit)
