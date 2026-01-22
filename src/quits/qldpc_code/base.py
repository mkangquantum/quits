"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np
import random
import networkx as nx
from scipy.linalg import circulant
from ..gf2_util import verify_css_logicals


class QldpcCode:
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

        pos = nx.get_node_attributes(self.graph, 'pos')
        if not draw_edges:
            nx.draw(self.graph, pos, node_color=self.node_colors, with_labels=True, font_color='white')
            return

        edges = self.graph.edges()
        edge_colors = [self.graph[u][v]['color'] for u, v in edges]
        self.graph.add_edges_from(edges)
        nx.draw(self.graph, pos, node_color=self.node_colors, edge_color=edge_colors, with_labels=True, font_color='white')
        return

    def build_graph(self):

        self.graph = nx.Graph()
        self.direction_inds = {'E': 0, 'N': 1, 'S': 2, 'W': 3}
        self.direction_colors = ['green', 'blue', 'orange', 'red']

        self.node_colors = []  # 'blue' for data qubits, 'green' for zcheck qubits, 'purple' for xcheck qubits
        self.edges = [[] for i in range(len(self.direction_inds))]  # edges of the Tanner graph of each direction

        self.rev_dics = [{} for i in range(len(self.direction_inds))]  # dictionaries used to efficiently construct the reversed Tanner graph for each direction
        self.rev_nodes = [[] for i in range(len(self.direction_inds))]  # nodes of the reversed Tanner graph of each direction
        self.rev_edges = [[] for i in range(len(self.direction_inds))]  # edges of the reversed Tanner graph of each direction.
        self.colored_edges = [{} for i in range(len(self.direction_inds))]  # for each direction, dictionary's key is the color, values are the edges
        self.num_colors = {direction: 0 for direction in self.direction_inds.keys()}
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

        self.edges[direction_ind] += [(control, target)]
        self.graph.add_edge(control, target, color=self.direction_colors[direction_ind])

        # add edge to rev graph
        self.rev_nodes[direction_ind] += [edge_no]
        if control not in self.rev_dics[direction_ind]:
            self.rev_dics[direction_ind][control] = [edge_no]
        else:
            self.rev_dics[direction_ind][control] += [edge_no]
        if target not in self.rev_dics[direction_ind]:
            self.rev_dics[direction_ind][target] = [edge_no]
        else:
            self.rev_dics[direction_ind][target] += [edge_no]
        return

    def color_edges(self):
        # Construct the reversed Tanner graph's edges from rev_dics dictionary
        for direction_ind in range(len(self.rev_edges)):
            dic = self.rev_dics[direction_ind]
            for nodes in dic.values():
                for i in range(len(nodes) - 1):
                    for j in range(i + 1, len(nodes)):
                        self.rev_edges[direction_ind] += [(nodes[i], nodes[j])]

        edge_colors = [[] for i in range(len(self.direction_inds))]  # list of colors of the reversed Tanner graph's nodes for each direction
        # Apply coloring to the reversed Tanner graph
        for direction_ind in range(len(self.rev_edges)):
            rev_graph = nx.Graph()
            rev_graph.add_nodes_from(self.rev_nodes[direction_ind])
            rev_graph.add_edges_from(self.rev_edges[direction_ind])

            edge_coloration = nx.greedy_color(rev_graph)
            # Somehow the dictionary returned by nx.greedy_color shuffles the keys (rev_nodes[direction_ind])
            # so the values (colors) need to be shuffled correctly.
            paired = list(zip(edge_coloration.keys(), edge_coloration.values()))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            _, reordered_colors = zip(*paired_sorted)
            edge_colors[direction_ind] = reordered_colors

        # Construct colored_edges (dictionary of edges of each direction and color)
        for direction_ind in range(len(self.colored_edges)):
            for i in range(len(self.edges[direction_ind])):
                edge = list(self.edges[direction_ind][i])
                color = edge_colors[direction_ind][i]

                if color not in self.colored_edges[direction_ind]:
                    self.colored_edges[direction_ind][color] = edge
                else:
                    self.colored_edges[direction_ind][color] += edge

        for direction in list(self.direction_inds.keys()):
            direction_ind = self.direction_inds[direction]
            self.num_colors[direction] = len(list(self.colored_edges[direction_ind].keys()))
        return


__all__ = ["QldpcCode"]
