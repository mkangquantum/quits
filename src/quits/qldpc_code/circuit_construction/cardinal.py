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

    # Draw the Tanner graph of the code.
    def draw_graph(self, draw_edges=True):
        code = self.code
        pos = nx.get_node_attributes(code.graph, 'pos')
        if not draw_edges:
            nx.draw(code.graph, pos, node_color=code.node_colors, with_labels=True, font_color='white')
            return

        edges = code.graph.edges()
        edge_colors = [code.graph[u][v]['color'] for u, v in edges]
        code.graph.add_edges_from(edges)
        nx.draw(code.graph, pos, node_color=code.node_colors, edge_color=edge_colors, with_labels=True, font_color='white')
        return

    def build_graph(self, **opts):
        code = self.code
        code.graph = nx.Graph()
        code.direction_colors = {'E': 'green', 'N': 'blue', 'S': 'orange', 'W': 'red'}

        code.node_colors = []  # 'blue' for data qubits, 'green' for zcheck qubits, 'purple' for xcheck qubits
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
        code.graph.add_edge(control, target, color=code.direction_colors[direction])
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
