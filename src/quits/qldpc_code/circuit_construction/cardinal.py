import random

import networkx as nx
import numpy as np

from .base import CircuitBuilder


class CardinalBuilder(CircuitBuilder):
    name = "cardinal"

    def build(self, code, **opts):
        self.build_graph(code, **opts)
        return code.graph

    # Draw the Tanner graph of the code.
    def draw_graph(self, code, draw_edges=True):
        pos = nx.get_node_attributes(code.graph, 'pos')
        if not draw_edges:
            nx.draw(code.graph, pos, node_color=code.node_colors, with_labels=True, font_color='white')
            return

        edges = code.graph.edges()
        edge_colors = [code.graph[u][v]['color'] for u, v in edges]
        code.graph.add_edges_from(edges)
        nx.draw(code.graph, pos, node_color=code.node_colors, edge_color=edge_colors, with_labels=True, font_color='white')
        return

    def build_graph(self, code, **opts):
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
    def add_edge(self, code, edge_no, direction_ind, control, target):
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

    def color_edges(self, code):
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
