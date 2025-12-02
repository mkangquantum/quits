"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np
import random
import networkx as nx
from scipy.linalg import circulant
from ldpc.mod2.mod2_numpy import nullspace
from .ldpc_utility import compute_lz_and_lx

# Parent class 
class QldpcCode:
    def __init__(self):

        self.hz, self.hx = None, None
        self.lz, self.lx = None, None

        self.data_qubits, self.zcheck_qubits, self.xcheck_qubits = None, None, None
        self.check_qubits, self.all_qubits = None, None
    
    def get_circulant_mat(self, size, power):
        return circulant(np.eye(size, dtype=int)[:,power])

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
                if h_base_placeholder[i,j] != 0:
                    h[i*lift_size:(i+1)*lift_size, j*lift_size:(j+1)*lift_size] = self.get_circulant_mat(lift_size, h_base[i,j])
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
                if h_base_placeholder[i,j] != 0:
                    hij_enc = h_base_enc[i,j]
                    if hij_enc == 0:                        
                        h[i*lift_size:(i+1)*lift_size, j*lift_size:(j+1)*lift_size] = self.get_circulant_mat(lift_size, 0)
                    else:
                        while hij_enc > 0:
                            power = hij_enc % lift_size
                            h[i*lift_size:(i+1)*lift_size, j*lift_size:(j+1)*lift_size] += self.get_circulant_mat(lift_size, power)
                            hij_enc = hij_enc // lift_size
        return h
    
    # Draw the Tanner graph of the code.
    def draw_graph(self, draw_edges=True):

        pos = nx.get_node_attributes(self.graph, 'pos')
        if not draw_edges:
            nx.draw(self.graph, pos, node_color=self.node_colors, with_labels=True, font_color='white')
            return

        edges = self.graph.edges()
        edge_colors = [self.graph[u][v]['color'] for u,v in edges]
        self.graph.add_edges_from(edges)
        nx.draw(self.graph, pos, node_color=self.node_colors, edge_color=edge_colors, with_labels=True, font_color='white')
        return

    def build_graph(self):

        self.graph = nx.Graph()
        self.direction_inds = {'E': 0, 'N': 1, 'S': 2, 'W': 3}
        self.direction_colors = ['green', 'blue', 'orange', 'red']

        self.node_colors = []                  # 'blue' for data qubits, 'green' for zcheck qubits, 'purple' for xcheck qubits
        self.edges = [[] for i in range(len(self.direction_inds))]          # edges of the Tanner graph of each direction  

        self.rev_dics = [{} for i in range(len(self.direction_inds))]       # dictionaries used to efficiently construct the reversed Tanner graph for each direction
        self.rev_nodes = [[] for i in range(len(self.direction_inds))]      # nodes of the reversed Tanner graph of each direction
        self.rev_edges = [[] for i in range(len(self.direction_inds))]      # edges of the reversed Tanner graph of each direction. 
        self.colored_edges = [{} for i in range(len(self.direction_inds))]  # for each direction, dictionary's key is the color, values are the edges 
        self.num_colors = {direction: 0 for direction in self.direction_inds.keys()}
        return   
    
    # Helper function for assigning bool to each edge of the classical code's parity check matrix 
    def get_classical_edge_bools(self, h, seed):

        c0_scores = {}
        c1_scores = {}
        edge_signs = {}
        random.seed(seed)

        for edge in np.argwhere(h==1):
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
                for i in range(len(nodes)-1):
                    for j in range(i+1, len(nodes)):
                        self.rev_edges[direction_ind] += [(nodes[i], nodes[j])]

        edge_colors = [[] for i in range(len(self.direction_inds))]    # list of colors of the reversed Tanner graph's nodes for each direction
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


# Hypergraph product (HGP) code
class HgpCode(QldpcCode):
    def __init__(self, h1, h2):
        '''
        :param h1: Parity check matrix of the first classical code used to construct the hgp code
        :param h2: Parity check matrix of the second classical code used to construct the hgp code
        '''
        super().__init__()

        self.h1, self.h2 = h1, h2    
        self.r1, self.n1 = h1.shape
        self.r2, self.n2 = h2.shape

        self.hz = np.concatenate((np.kron(h2, np.eye(self.n1, dtype=int)), 
                                  np.kron(np.eye(self.r2, dtype=int), h1.T)), axis=1)
        self.hx = np.concatenate((np.kron(np.eye(self.n2, dtype=int), h1), 
                                  np.kron(h2.T, np.eye(self.r1, dtype=int))), axis=1)
        
        self.l1 = nullspace(self.h1)
        self.l2 = nullspace(self.h2)
        self.k1, self.k2 = self.l1.shape[0], self.l2.shape[0]

        if self.r1 == self.n1 - self.k1 and self.r2 == self.n2 - self.k2:   # If both classical parity check matrices are full-rank
            self.lz, self.lx = self.get_logicals()                          # set logical operators in the "canonical form"
        else:
            self.lz, self.lx = compute_lz_and_lx(self.hx, self.hz)

    def get_logicals(self):
        '''
        :return: Logical operators of the code as a list of tuples (logical_z, logical_x)
                 where logical_z and logical_x are numpy arrays of shape (num_logicals, num_data_qubits)
                 The logicals are written in the "canonical form" as described in arXiv:2204.10812
        '''
        lz = np.zeros((self.k1*self.k2, self.hz.shape[1]), dtype=int)
        lx = np.zeros((self.k1*self.k2, self.hx.shape[1]), dtype=int)

        cnt = 0
        for i in range(self.k2):
            ei = np.zeros(self.h2.shape[1], dtype=int)
            ei[i] = 1
            for j in range(self.k1):
                ej = np.zeros(self.h1.shape[1], dtype=int)
                ej[j] = 1

                lz[cnt,:self.n1*self.n2] = np.kron(ei, self.l1[j,:])
                lx[cnt,:self.n1*self.n2] = np.kron(self.l2[i,:], ej)

                cnt += 1

        return lz, lx

    def build_graph(self, seed=1):

        super().build_graph()
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                node = i + j * (self.n1 + self.r1)
                data_qubits += [node]               
                self.graph.add_node(node, pos=(i, j))
                self.node_colors += ['blue']

        start = self.n1
        for i in range(self.r1):
            for j in range(self.n2):
                node = start + i + j * (self.n1 + self.r1)
                xcheck_qubits += [node]               
                self.graph.add_node(node, pos=(i+self.n1, j))
                self.node_colors += ['purple']
                
        start = self.n2 * (self.n1 + self.r1)
        for i in range(self.n1):
            for j in range(self.r2):
                node = start + i + j * (self.n1 + self.r1)
                zcheck_qubits += [node]                
                self.graph.add_node(node, pos=(i, j+self.n2))
                self.node_colors += ['green']
                
        start = self.n2 * (self.n1 + self.r1) + self.n1
        for i in range(self.r1):
            for j in range(self.r2):
                node = start + i + j * (self.n1 + self.r1)
                data_qubits += [node]                
                self.graph.add_node(node, pos=(i+self.n1, j+self.n2))
                self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))

        hedge_bool_list = self.get_classical_edge_bools(self.h1, seed)
        vedge_bool_list = self.get_classical_edge_bools(self.h2, seed)
    
        edge_no = 0
        for classical_edge in np.argwhere(self.h1==1):
            c0, c1 = classical_edge
            edge_bool = hedge_bool_list[(c0, c1)]
            for k in range(self.n2 + self.r2):
                control, target = (k*(self.n1 + self.r1) + c0+self.n1, k*(self.n1 + self.r1) + c1)       
                if (k < self.n2) ^ edge_bool:
                    direction_ind = self.direction_inds['E']
                else:
                    direction_ind = self.direction_inds['W']
                self.add_edge(edge_no, direction_ind, control, target)
                edge_no += 1

        for classical_edge in np.argwhere(self.h2==1):
            c0, c1 = classical_edge
            edge_bool = vedge_bool_list[(c0, c1)]
            for k in range(self.n1 + self.r1):
                control, target = (k + c1*(self.n1 + self.r1), k + (c0+self.n2)*(self.n1 + self.r1))
                if (k < self.n1) ^ edge_bool:
                    direction_ind = self.direction_inds['N']
                else:
                    direction_ind = self.direction_inds['S']
                self.add_edge(edge_no, direction_ind, control, target)
                edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return


# Quasi-cyclic lifted product (QLP) code
class QlpCode(QldpcCode):
    def __init__(self, b1, b2, lift_size):
        '''
        :param b1: First base matrix used to construct the lp code. Each entry is the power of the monomial. 
                   e.g. b1 = np.array([[0, 0], [0, 3]]) represents the matrix of monomials [[1, 1], [1, x^3]].
        :param b2: Second base matrix used to construct the lp code. Each entry is the power of the monomial.
        :param lift_size: Size of cyclic matrix to which each monomial entry is lifted. 
        '''
        super().__init__()

        self.b1, self.b2 = b1, b2
        self.lift_size = lift_size
        self.m1, self.n1 = b1.shape
        self.m2, self.n2 = b2.shape

        b1T = (self.lift_size - b1).T % self.lift_size
        b2T = (self.lift_size - b2).T % self.lift_size
        b1_placeholder = np.ones(b1.shape, dtype=int)
        b2_placeholder = np.ones(b2.shape, dtype=int)

        hz_base = np.concatenate((np.kron(b2, np.eye(self.n1, dtype=int)), 
                                  np.kron(np.eye(self.m2, dtype=int), b1T)), axis=1)
        hx_base = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1), 
                                  np.kron(b2T, np.eye(self.m1, dtype=int))), axis=1)
        hz_base_placeholder = np.concatenate((np.kron(b2_placeholder, np.eye(self.n1, dtype=int)), 
                                              np.kron(np.eye(self.m2, dtype=int), b1_placeholder.T)), axis=1)
        hx_base_placeholder = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1_placeholder), 
                                              np.kron(b2_placeholder.T, np.eye(self.m1, dtype=int))), axis=1)
        
        self.hz = self.lift(self.lift_size, hz_base, hz_base_placeholder)
        self.hx = self.lift(self.lift_size, hx_base, hx_base_placeholder)
        self.lz, self.lx = compute_lz_and_lx(self.hx, self.hz)

    def build_graph(self, seed=1):

        super().build_graph()
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    data_qubits += [node]               
                    self.graph.add_node(node, pos=(i, j))
                    self.node_colors += ['blue']

        start = self.n1 * self.lift_size
        for i in range(self.m1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    xcheck_qubits += [node]               
                    self.graph.add_node(node, pos=(i+self.n1, j))
                    self.node_colors += ['purple']                    
                    
        start = self.n2 * (self.n1 + self.m1) * self.lift_size
        for i in range(self.n1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    zcheck_qubits += [node]                
                    self.graph.add_node(node, pos=(i, j+self.n2))
                    self.node_colors += ['green']

        start = (self.n2 * (self.n1 + self.m1) + self.n1) * self.lift_size        
        for i in range(self.m1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    data_qubits += [node]                
                    self.graph.add_node(node, pos=(i+self.n1, j+self.n2))
                    self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))   

        hedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)
        vedge_bool_list = self.get_classical_edge_bools(np.ones(self.b2.shape, dtype=int), seed)
    
        edge_no = 0
        for i in range(self.m1):
            for j in range(self.n1):
                shift = self.b1[i, j]
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n2 + self.m2):
                        if (k < self.n2) ^ edge_bool:
                            direction_ind = self.direction_inds['E']     
                        else:
                            direction_ind = self.direction_inds['W']                                                 

                        control = (k * (self.n1+self.m1) + self.n1 + i) * self.lift_size + (l + shift) % self.lift_size
                        target = (k * (self.n1+self.m1) + j) * self.lift_size + l
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        for i in range(self.m2):
            for j in range(self.n2):
                shift = self.b2[i, j]
                edge_bool = vedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n1 + self.m1):
                        if (k < self.n1) ^ edge_bool:
                            direction_ind = self.direction_inds['N']     
                        else:
                            direction_ind = self.direction_inds['S']   

                        control = (k + j * (self.n1 + self.m1)) * self.lift_size + l
                        target = (k + (i + self.n2) * (self.n1 + self.m1)) * self.lift_size + (l + shift) % self.lift_size
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return
    

# Quasi-cyclic lifted product (QLP) code with polynomial entries in the base matrices
class QlpCode2(QldpcCode):
    def __init__(self, b1, b2, lift_size):
        '''
        :param b1: First base matrix used to construct the lp code. Each entry is the list of powers of the polynomial terms. 
                   e.g. b1 = [[[0], [0,1], []], [[], [0], [0,1]]] represents the matrix of monomials [[1, 1+x, 0], [0, 1, 1+x]].
        :param b2: Second base matrix used to construct the lp code. Each entry is the list of powers of the polynomial terms.
        :param lift_size: Size of cyclic matrix to which each polynomial term is lifted. 
        '''
        super().__init__()

        self.b1, self.b2 = b1, b2
        self.lift_size = lift_size

        self.m1, self.n1 = len(b1), len(b1[0])
        self.m2, self.n2 = len(b2), len(b2[0])

        # Base matrices where each entry ENCODEs the powers of polynomial terms in base of lift_size
        b1_enc = np.zeros((self.m1, self.n1), dtype=int)
        b1T_enc = np.zeros((self.n1, self.m1), dtype=int)
        b2_enc = np.zeros((self.m2, self.n2), dtype=int)
        b2T_enc = np.zeros((self.n2, self.m2), dtype=int)
        self.b1_placeholder = np.zeros((self.m1, self.n1), dtype=int)
        self.b2_placeholder = np.zeros((self.m2, self.n2), dtype=int)

        for i in range(self.m1):
            for j in range(self.n1):
                if self.b1[i][j] == []:
                    continue
                self.b1_placeholder[i,j] = 1

                bij, bTij = 0, 0
                for k in range(len(self.b1[i][j])):
                    bij += self.lift_size**k * self.b1[i][j][k] 
                    bTij += self.lift_size**k * ((self.lift_size - self.b1[i][j][k]) % self.lift_size)
                b1_enc[i,j] = bij
                b1T_enc[j,i] = bTij 

        for i in range(self.m2):
            for j in range(self.n2):
                if self.b2[i][j] == []:
                    continue
                self.b2_placeholder[i,j] = 1

                bij, bTij = 0, 0
                for k in range(len(self.b2[i][j])):
                    bij += self.lift_size**k * self.b2[i][j][k] 
                    bTij += self.lift_size**k * ((self.lift_size - self.b2[i][j][k]) % self.lift_size)
                b2_enc[i,j] = bij
                b2T_enc[j,i] = bTij 

        hz_base_enc = np.concatenate((np.kron(b2_enc, np.eye(self.n1, dtype=int)), 
                                  np.kron(np.eye(self.m2, dtype=int), b1T_enc)), axis=1)
        hx_base_enc = np.concatenate((np.kron(np.eye(self.n2, dtype=int), b1_enc), 
                                  np.kron(b2T_enc, np.eye(self.m1, dtype=int))), axis=1)
        hz_base_placeholder = np.concatenate((np.kron(self.b2_placeholder, np.eye(self.n1, dtype=int)), 
                                              np.kron(np.eye(self.m2, dtype=int), self.b1_placeholder.T)), axis=1)
        hx_base_placeholder = np.concatenate((np.kron(np.eye(self.n2, dtype=int), self.b1_placeholder), 
                                              np.kron(self.b2_placeholder.T, np.eye(self.m1, dtype=int))), axis=1)
        
        self.hz = self.lift_enc(self.lift_size, hz_base_enc, hz_base_placeholder)
        self.hx = self.lift_enc(self.lift_size, hx_base_enc, hx_base_placeholder)
        self.lz, self.lx = compute_lz_and_lx(self.hx, self.hz)

    def build_graph(self, seed=1):

        super().build_graph()
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.n1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    data_qubits += [node]               
                    self.graph.add_node(node, pos=(i, j))
                    self.node_colors += ['blue']

        start = self.n1 * self.lift_size
        for i in range(self.m1):
            for j in range(self.n2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    xcheck_qubits += [node]               
                    self.graph.add_node(node, pos=(i+self.n1, j))
                    self.node_colors += ['purple']                    
                    
        start = self.n2 * (self.n1 + self.m1) * self.lift_size
        for i in range(self.n1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    zcheck_qubits += [node]                
                    self.graph.add_node(node, pos=(i, j+self.n2))
                    self.node_colors += ['green']

        start = (self.n2 * (self.n1 + self.m1) + self.n1) * self.lift_size        
        for i in range(self.m1):
            for j in range(self.m2):
                for l in range(self.lift_size):
                    node = start + (i + j * (self.n1 + self.m1)) * self.lift_size + l 
                    data_qubits += [node]                
                    self.graph.add_node(node, pos=(i+self.n1, j+self.n2))
                    self.node_colors += ['blue']

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))   

        hedge_bool_list = self.get_classical_edge_bools(self.b1_placeholder, seed)
        vedge_bool_list = self.get_classical_edge_bools(self.b2_placeholder, seed)
    
        edge_no = 0
        for i in range(self.m1):
            for j in range(self.n1):
                if self.b1_placeholder[i,j] == 0:
                    continue
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n2 + self.m2):
                        if (k < self.n2) ^ edge_bool:
                            direction_ind = self.direction_inds['E']     
                        else:
                            direction_ind = self.direction_inds['W']                                                 

                        for shift in self.b1[i][j]:
                            control = (k * (self.n1+self.m1) + self.n1 + i) * self.lift_size + (l + shift) % self.lift_size
                            target = (k * (self.n1+self.m1) + j) * self.lift_size + l
                            self.add_edge(edge_no, direction_ind, control, target)
                            edge_no += 1

        for i in range(self.m2):
            for j in range(self.n2):
                if self.b2_placeholder[i,j] == 0:
                    continue
                edge_bool = vedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(self.n1 + self.m1):
                        if (k < self.n1) ^ edge_bool:
                            direction_ind = self.direction_inds['N']     
                        else:
                            direction_ind = self.direction_inds['S']   

                        for shift in self.b2[i][j]:
                            control = (k + j * (self.n1 + self.m1)) * self.lift_size + l
                            target = (k + (i + self.n2) * (self.n1 + self.m1)) * self.lift_size + (l + shift) % self.lift_size
                            self.add_edge(edge_no, direction_ind, control, target)
                            edge_no += 1

        # Color the edges of self.graph
        self.color_edges()
        return    
    

# Balanced product cyclic (BPC) code
# To precisely match arXiv:2411.03302, you should insert $p_2^T$ from the paper into $p_2$ here. 
# That is, for the second polynomial, the entries should be lift_size minus the powers in arXiv:2411.03302.
# This is due to the different convention of the parity check matrix we use in QUITS. 
class BpcCode(QldpcCode):
    def __init__(self, p1, p2, lift_size, factor):
        '''
        :param p1: First polynomial used to construct the bp code. Each entry of the list is the power of each polynomial term. 
                   e.g. p1 = [0, 1, 5] represents the polynomial 1 + x + x^5
        :param p2: Second polynomial used to construct the bp code. Each entry of the list is the power of each polynomial term. 
        :param lift_size: Size of cyclic matrix to which each monomial entry is lifted. 
        :param factor: Power of the monomial generator of the cyclic subgroup that is factored out by the balanced product. 
                       e.g. if factor == 3, cyclic subgroup <x^3> is factored out. 
        '''
        super().__init__()

        self.p1, self.p2 = p1, p2
        self.lift_size = lift_size
        self.factor = factor

        b1 = np.zeros((self.factor, self.factor), dtype=int)
        b1_placeholder = np.zeros((self.factor, self.factor), dtype=int)
        for power in p1:
            mat, mat_placeholder = self.get_block_mat(power)
            b1 = b1 + mat
            b1_placeholder = b1_placeholder + mat_placeholder
        b1T = (self.lift_size - b1.T) % self.lift_size
        b1T_placeholder = b1_placeholder.T
        
        self.b1, self.b1T = b1, b1T
        self.b1_placeholder, self.b1T_placeholder = b1_placeholder, b1T_placeholder

        h1 = self.lift(self.lift_size, b1, b1_placeholder)
        h1T = self.lift(self.lift_size, b1T, b1T_placeholder)

        h2 = np.zeros((self.lift_size, self.lift_size), dtype=int)
        for power in p2:
            h2 = h2 + self.get_circulant_mat(self.lift_size, power)
        h2 = np.kron(np.eye(self.factor, dtype=int), h2)
        h2T = h2.T

        self.hz = np.concatenate((h2, h1T), axis=1)
        self.hx = np.concatenate((h1, h2T), axis=1)
        self.lz, self.lx = self.get_logicals()    # logical operators in the "canonical form"

    def get_block_mat(self, power):
        gen_mat = self.get_circulant_mat(self.factor, 1)
        gen_mat[0,-1] = 2

        mat = np.linalg.matrix_power(gen_mat, power)
        mat_placeholder = (mat > 0) * 1

        mat = np.log2(mat + 1e-8).astype(int)
        mat = mat * mat_placeholder * self.factor
        return mat, mat_placeholder
    
    def get_logicals(self):
        '''
        :return: Logical operators of the code as a list of tuples (logical_z, logical_x)
                 where logical_z and logical_x are numpy arrays of shape (num_logicals, num_data_qubits)
                 The logicals are written in the "canonical form" as described in Eq. 30 of arXiv:2411.03302
        '''

        lz = np.zeros((2*(self.factor-1)**2, self.hz.shape[1]), dtype=int)
        lx = np.zeros((2*(self.factor-1)**2, self.hx.shape[1]), dtype=int)

        cnt = 0
        for i in range(self.factor-1):
            for j in range(self.factor-1):
                yi_vec = self.get_circulant_mat(self.factor, 0)[:,i]
                xjgx_vec = (self.get_circulant_mat(self.factor, 0) + self.get_circulant_mat(self.factor, 1))[:,j]
                xjgx_vec = np.tile(xjgx_vec, self.lift_size//self.factor)

                prod = np.kron(yi_vec, xjgx_vec)
                lz[cnt,:] = np.concatenate((np.zeros(self.hz.shape[1]-len(prod), dtype=int), prod))
                lx[cnt,:] = np.concatenate((prod, np.zeros(self.hx.shape[1]-len(prod), dtype=int)))

                cnt += 1

        for i in range(self.factor-1):
            for j in range(self.factor-1):
                yigy_vec = (self.get_circulant_mat(self.factor, 0) + self.get_circulant_mat(self.factor, 1))[:,i]
                xj_vec = self.get_circulant_mat(self.factor, 0)[:,j]      
                xj_vec = np.tile(xj_vec, self.lift_size//self.factor)

                prod = np.kron(yigy_vec, xj_vec)
                lz[cnt,:] = np.concatenate((prod, np.zeros(self.hz.shape[1]-len(prod), dtype=int)))
                lx[cnt,:] = np.concatenate((np.zeros(self.hx.shape[1]-len(prod), dtype=int), prod))
                
                cnt += 1

        return lz, lx    

    def build_graph(self, seed=1):

        super().build_graph()
        data_qubits, zcheck_qubits, xcheck_qubits = [], [], []

        # Add nodes to the Tanner graph
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = i * self.lift_size + l
                data_qubits += [node]
                self.graph.add_node(node, pos=(2*i, 0))
                self.node_colors += ['blue']

        start = self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                xcheck_qubits += [node] 
                self.graph.add_node(node, pos=(2*i+1, 0))
                self.node_colors += ['purple']
                    
        start = 2 * self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                zcheck_qubits += [node] 
                self.graph.add_node(node, pos=(2*i, 1))
                self.node_colors += ['green']

        start = 3 * self.factor * self.lift_size
        for i in range(self.factor):
            for l in range(self.lift_size):
                node = start + i * self.lift_size + l
                data_qubits += [node]
                self.graph.add_node(node, pos=(2*i+1, 1))
                self.node_colors += ['blue']             

        self.data_qubits = sorted(np.array(data_qubits))
        self.zcheck_qubits = sorted(np.array(zcheck_qubits))
        self.xcheck_qubits = sorted(np.array(xcheck_qubits))
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = sorted(np.array(data_qubits + zcheck_qubits + xcheck_qubits))   

        hedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)
        vedge_bool_list = self.get_classical_edge_bools(np.ones(self.b1.shape, dtype=int), seed)        

        # Add edges to the Tanner graph of each direction
        edge_no = 0
        for i in range(self.factor):          
            for j in range(self.factor):   
                shift = self.b1[i,j] 
                edge_bool = hedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(2):  # 0 : bottom, 1 : top              
                        if k ^ edge_bool:
                            direction_ind = self.direction_inds['E']
                        else:
                            direction_ind = self.direction_inds['W']

                        control = (2*k+1)*self.factor*self.lift_size + i*self.lift_size + (l + shift) % self.lift_size
                        target = 2*k*self.factor*self.lift_size + j*self.lift_size + l
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1

        def shuffle(node_no, qubit_no):
            m, r = qubit_no // self.factor, qubit_no % self.factor
            return r, self.lift_size // self.factor * node_no + m

        for i in range(self.factor):          
            for j in range(len(self.p2)):   
                shift = self.p2[j]

                for l in range(self.lift_size):
                    for k in range(2):  # 0 : left, 1 : right   
                        i_shuffled, _ = shuffle(i, l)
                        j_shuffled, _ = shuffle(i, (l + shift) % self.lift_size)    
                        edge_bool = vedge_bool_list[(i_shuffled, j_shuffled)]
                        if k ^ edge_bool:
                            direction_ind = self.direction_inds['N']
                        else:
                            direction_ind = self.direction_inds['S']

                        control = k*self.factor*self.lift_size + i*self.lift_size + l
                        target = (2+k)*self.factor*self.lift_size + i*self.lift_size + (l + shift) % self.lift_size
                        self.add_edge(edge_no, direction_ind, control, target)
                        edge_no += 1                                          

        # Color the edges of self.graph
        self.color_edges()
        return

