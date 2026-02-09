import random

import networkx as nx
import numpy as np

from ...circuit import Circuit
from ...noise import ErrorModel
from .circuit_build_options import CircuitBuildOptions
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
        code.direction_inds = {'E': 0, 'N': 1, 'S': 2, 'W': 3}
        code.direction_colors = ['green', 'blue', 'orange', 'red']

        code.node_colors = []  # 'blue' for data qubits, 'green' for zcheck qubits, 'purple' for xcheck qubits
        code.edges = [[] for _ in range(len(code.direction_inds))]  # edges of the Tanner graph of each direction

        # dictionaries used to efficiently construct the reversed Tanner graph for each direction
        code.rev_dics = [{} for _ in range(len(code.direction_inds))]
        code.rev_nodes = [[] for _ in range(len(code.direction_inds))]  # nodes of the reversed Tanner graph of each direction
        code.rev_edges = [[] for _ in range(len(code.direction_inds))]  # edges of the reversed Tanner graph of each direction.
        code.colored_edges = [{} for _ in range(len(code.direction_inds))]  # for each direction, dictionary's key is the color, values are the edges
        code.num_colors = {direction: 0 for direction in code.direction_inds.keys()}
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
    def add_edge(self, edge_no, direction_ind, control, target):
        code = self.code
        code.edges[direction_ind] += [(control, target)]
        code.graph.add_edge(control, target, color=code.direction_colors[direction_ind])

        # add edge to rev graph
        code.rev_nodes[direction_ind] += [edge_no]
        if control not in code.rev_dics[direction_ind]:
            code.rev_dics[direction_ind][control] = [edge_no]
        else:
            code.rev_dics[direction_ind][control] += [edge_no]
        if target not in code.rev_dics[direction_ind]:
            code.rev_dics[direction_ind][target] = [edge_no]
        else:
            code.rev_dics[direction_ind][target] += [edge_no]
        return

    def color_edges(self):
        code = self.code
        # Construct the reversed Tanner graph's edges from rev_dics dictionary
        for direction_ind in range(len(code.rev_edges)):
            dic = code.rev_dics[direction_ind]
            for nodes in dic.values():
                for i in range(len(nodes) - 1):
                    for j in range(i + 1, len(nodes)):
                        code.rev_edges[direction_ind] += [(nodes[i], nodes[j])]

        # list of colors of the reversed Tanner graph's nodes for each direction
        edge_colors = [[] for _ in range(len(code.direction_inds))]
        # Apply coloring to the reversed Tanner graph
        for direction_ind in range(len(code.rev_edges)):
            rev_graph = nx.Graph()
            rev_graph.add_nodes_from(code.rev_nodes[direction_ind])
            rev_graph.add_edges_from(code.rev_edges[direction_ind])

            edge_coloration = nx.greedy_color(rev_graph)
            # Somehow the dictionary returned by nx.greedy_color shuffles the keys (rev_nodes[direction_ind])
            # so the values (colors) need to be shuffled correctly.
            paired = list(zip(edge_coloration.keys(), edge_coloration.values()))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            _, reordered_colors = zip(*paired_sorted)
            edge_colors[direction_ind] = reordered_colors

        # Construct colored_edges (dictionary of edges of each direction and color)
        for direction_ind in range(len(code.colored_edges)):
            for i in range(len(code.edges[direction_ind])):
                edge = list(code.edges[direction_ind][i])
                color = edge_colors[direction_ind][i]

                if color not in code.colored_edges[direction_ind]:
                    code.colored_edges[direction_ind][color] = edge
                else:
                    code.colored_edges[direction_ind][color] += edge

        for direction in list(code.direction_inds.keys()):
            direction_ind = code.direction_inds[direction]
            code.num_colors[direction] = len(list(code.colored_edges[direction_ind].keys()))
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
        :return circuit: String that can be converted to Stim circuit by stim.Circuit()
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
        directions = list(code.direction_inds.keys())
        circ = Circuit(code.all_qubits)

        def _add_stabilizer_round():
            circ.add_hadamard_layer(code.xcheck_qubits)
            for direction_ind in range(len(directions)):
                direction = directions[direction_ind]
                for color in range(code.num_colors[direction]):
                    edges = code.colored_edges[direction_ind][color]
                    circ.add_cnot_layer(edges)
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

        circ.set_error_model(error_model)
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

        return circ.circuit
