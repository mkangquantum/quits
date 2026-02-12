import random

import networkx as nx
import numpy as np
import stim

from ...circuit import Circuit
from ...noise import ErrorModel
from .circuit_build_options import CircuitBuildOptions
from .edge_coloration import edge_coloration
from .base import CircuitBuilder


class CardinalBuilder(CircuitBuilder):
    name = "cardinal"

    def __init__(self, code=None):
        self.code = code

    def build(self, code, **opts):
        self.code = code
        self.build_graph(**opts)
        return self.code.graph

    # Draw the Tanner graph node-only, direction, or layer-color view.
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
        code = self.code
        part = part.lower()
        if part not in ("node", "color", "direction"):
            raise ValueError("For cardinal draw_graph, part must be one of: node, color, direction.")
        if code is None or not hasattr(code, "graph"):
            raise ValueError("CardinalBuilder.draw_graph requires an initialized code graph.")

        graph = code.graph
        pos = nx.get_node_attributes(graph, "pos")
        if x_scale != 1.0 or y_scale != 1.0:
            pos = {k: (v[0] * x_scale, v[1] * y_scale) for k, v in pos.items()}

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
        node_colors = [_node_color(node) for node in nodes]

        if figsize is not None:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=figsize)
            except ImportError:
                pass

        if part == "node":
            # draw_edges is kept for API compatibility with ZXColorationBuilder.draw_graph.
            nx.draw(
                graph,
                pos,
                nodelist=nodes,
                node_color=node_colors,
                edgelist=[],
                node_size=node_size,
                font_size=font_size,
                with_labels=True,
                font_color="white",
            )
            return

        if part == "direction":
            direction_color_map = {
                "E": "tab:green",
                "N": "tab:blue",
                "S": "tab:orange",
                "W": "tab:red",
            }
            direction_by_edge = {}
            for direction in ("E", "N", "S", "W"):
                for u, v in getattr(code, f"edges_{direction}", []):
                    direction_by_edge[frozenset((u, v))] = direction

            edges = list(graph.edges())
            edge_colors = [
                direction_color_map.get(direction_by_edge.get(frozenset((u, v))), "tab:gray")
                for u, v in edges
            ]

            nx.draw(
                graph,
                pos,
                nodelist=nodes,
                node_color=node_colors,
                edgelist=edges,
                edge_color=edge_colors,
                node_size=node_size,
                font_size=font_size,
                with_labels=True,
                font_color="white",
            )
            return

        # part == "color": color each edge by its CNOT layer in the cardinal schedule.
        palette = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan", "gold", "navy",
            "teal", "crimson", "darkorange", "slateblue",
            "seagreen", "indigo", "peru", "darkcyan",
            "firebrick", "darkgreen", "sienna", "dodgerblue",
        ]

        layer_index_by_edge = {}
        layer = 0
        direction_order = ("E", "N", "S", "W")
        colored_edges_by_direction = {
            "E": getattr(code, "colored_edges_E", {}),
            "N": getattr(code, "colored_edges_N", {}),
            "S": getattr(code, "colored_edges_S", {}),
            "W": getattr(code, "colored_edges_W", {}),
        }
        num_colors = getattr(code, "num_colors", {"E": 0, "N": 0, "S": 0, "W": 0})
        for direction in direction_order:
            for color in range(num_colors.get(direction, 0)):
                for u, v in colored_edges_by_direction[direction].get(color, []):
                    layer_index_by_edge[frozenset((u, v))] = layer
                layer += 1

        edges = list(graph.edges())
        edge_colors = [
            palette[layer_index_by_edge.get(frozenset((u, v)), 0) % len(palette)]
            for u, v in edges
        ]

        nx.draw(
            graph,
            pos,
            nodelist=nodes,
            node_color=node_colors,
            edgelist=edges,
            edge_color=edge_colors,
            node_size=node_size,
            font_size=font_size,
            with_labels=True,
            font_color="white",
        )
        return

    def build_graph(self, **opts):
        code = self.code
        code.graph = nx.Graph()

        code.edges_E = []  # edges of the Tanner graph of east direction
        code.edges_N = []  # edges of the Tanner graph of north direction
        code.edges_S = []  # edges of the Tanner graph of south direction
        code.edges_W = []  # edges of the Tanner graph of west direction

        code.colored_edges_E = {}  # for east direction, key is color, values are edges
        code.colored_edges_N = {}  # for north direction, key is color, values are edges
        code.colored_edges_S = {}  # for south direction, key is color, values are edges
        code.colored_edges_W = {}  # for west direction, key is color, values are edges
        code.num_colors = {'E': 0, 'N': 0, 'S': 0, 'W': 0}
        return

    # Helper function for assigning bool to each edge of the classical code's parity check matrix
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

    # Helper function for adding edges
    def add_edge(self, direction, control, target):
        code = self.code
        edge = (control, target)
        if direction == 'E':
            code.edges_E += [edge]
        elif direction == 'N':
            code.edges_N += [edge]
        elif direction == 'S':
            code.edges_S += [edge]
        elif direction == 'W':
            code.edges_W += [edge]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        code.graph.add_edge(control, target)
        return

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
        code.colored_edges_N = edge_coloration(_build_direction_graph(code.edges_N))
        code.colored_edges_S = edge_coloration(_build_direction_graph(code.edges_S))
        code.colored_edges_W = edge_coloration(_build_direction_graph(code.edges_W))

        code.num_colors['E'] = len(code.colored_edges_E)
        code.num_colors['N'] = len(code.colored_edges_N)
        code.num_colors['S'] = len(code.colored_edges_S)
        code.num_colors['W'] = len(code.colored_edges_W)
        # Total number of entangling gate layers across all edge-color groups.
        code.depth = sum(code.num_colors.values())
        return

    def get_cardinal_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis='Z',
        circuit_build_options=None,
    ):
        '''
        Returns the full Stim circuit for circuit-level simulations.
        Errors occur at each and every gate.

        :param error_model: ErrorModel that stores idle/single-/two-qubit/spam error parameters.
        :param num_rounds: Number of stabilizer measurement rounds
        :param basis: Basis in which logical codewords are stored. Options are 'Z' and 'X'.
        :param circuit_build_options: CircuitBuildOptions for detector/noisy-round/final-meas behavior.
        :return circuit: Stim circuit
        '''
        code = self.code
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")

        basis = basis.upper()
        if basis not in ('Z', 'X'):
            raise ValueError("basis must be 'Z' or 'X'")
        get_Z_detectors = True if basis == 'Z' or circuit_build_options.get_all_detectors else False
        get_X_detectors = True if basis == 'X' or circuit_build_options.get_all_detectors else False
        directions = ['E', 'N', 'S', 'W']
        circ = Circuit(code.all_qubits)

        def _add_stabilizer_round():
            def _flatten_edge_pairs(edge_pairs):
                return [q for control, target in edge_pairs for q in (control, target)]

            colored_edges_by_direction = {
                'E': code.colored_edges_E,
                'N': code.colored_edges_N,
                'S': code.colored_edges_S,
                'W': code.colored_edges_W,
            }
            circ.add_hadamard_layer(code.xcheck_qubits)
            for direction_ind in range(len(directions)):
                direction = directions[direction_ind]
                for color in range(code.num_colors[direction]):
                    edges = colored_edges_by_direction[direction][color]
                    circ.add_cnot_layer(_flatten_edge_pairs(edges))
            circ.add_hadamard_layer(code.xcheck_qubits)
            circ.add_measure_reset_layer(code.check_qubits)
            return

        ################## Logical state prep ##################
        if circuit_build_options.noisy_zeroth_round:
            circ.set_error_model(error_model)
        else:
            circ.set_error_model(ErrorModel.zero())
            
        circ.add_reset(code.data_qubits, basis)
        circ.add_reset(code.check_qubits)
        circ.add_tick()

        _add_stabilizer_round()

        if basis == 'Z':
            for i in range(1, len(code.zcheck_qubits)+1)[::-1]:
                circ.add_detector([len(code.xcheck_qubits) + i])
        elif basis == 'X':
            for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
                circ.add_detector([i])

        ############## Logical memory w/ noise ###############
        circ.set_error_model(error_model)

        if num_rounds > 0:
            circ.start_loop(num_rounds)

            _add_stabilizer_round()

            if get_Z_detectors:
                for i in range(1, len(code.zcheck_qubits)+1)[::-1]:
                    ind = len(code.xcheck_qubits) + i
                    circ.add_detector([ind, ind + len(code.check_qubits)])
            if get_X_detectors:
                for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
                    circ.add_detector([i, i + len(code.check_qubits)])

            circ.end_loop()

        ################## Logical measurement ##################
        if not circuit_build_options.noisy_final_meas:
            circ.set_error_model(ErrorModel.zero())

        circ.add_measure(code.data_qubits, basis)

        if basis == 'Z':
            for i in range(1, len(code.zcheck_qubits)+1)[::-1]:
                inds = np.array([len(code.data_qubits) + len(code.xcheck_qubits) + i])
                inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hz[len(code.zcheck_qubits)-i, :]==1)[0]))
                circ.add_detector(inds)

            for i in range(len(code.lz)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lz[i,:]==1)[0])

        elif basis == 'X':
            for i in range(1, len(code.xcheck_qubits)+1)[::-1]:
                inds = np.array([len(code.data_qubits) + i])
                inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hx[len(code.xcheck_qubits)-i, :]==1)[0]))
                circ.add_detector(inds)

            for i in range(len(code.lx)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lx[i,:]==1)[0])

        return stim.Circuit(circ.circuit)
