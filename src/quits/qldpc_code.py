"""
@author: Mingyu Kang, Yingjia Lin
"""

import numpy as np
import random
import networkx as nx
from .ldpc_utility import compute_lz_and_lx
from scipy.linalg import circulant

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
        h = np.zeros((h_base.shape[0] * lift_size, h_base.shape[1] * lift_size), dtype=int)
        for i in range(h_base.shape[0]):
            for j in range(h_base.shape[1]):
                if h_base_placeholder[i,j] != 0:
                    h[i*lift_size:(i+1)*lift_size, j*lift_size:(j+1)*lift_size] = self.get_circulant_mat(lift_size, h_base[i,j])
        return h
    
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
            edge_colors[direction_ind] = list(nx.greedy_color(rev_graph).values())

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
        
        self.lz, self.lx = compute_lz_and_lx(self.hx, self.hz)

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


# Balanced product cyclic (BPC) code
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
        self.lz, self.lx = compute_lz_and_lx(self.hx, self.hz)

    def get_block_mat(self, power):
        gen_mat = self.get_circulant_mat(self.factor, 1)
        gen_mat[0,-1] = 2

        mat = np.linalg.matrix_power(gen_mat, power)
        mat_placeholder = (mat > 0) * 1

        mat = np.log2(mat + 1e-8).astype(int)
        mat = mat * mat_placeholder * self.factor
        return mat, mat_placeholder

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

        for i in range(self.factor):          
            for j in range(self.factor):   
                shift = self.p2[j]
                edge_bool = vedge_bool_list[(i, j)]

                for l in range(self.lift_size):
                    for k in range(2):  # 0 : left, 1 : right              
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

    