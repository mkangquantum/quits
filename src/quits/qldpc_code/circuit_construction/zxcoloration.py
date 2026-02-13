import networkx as nx
import numpy as np
import stim

# Reference: Q. Xu et al., arXiv:2308.08648 
from ...circuit import Circuit
from ...noise import ErrorModel
from .base import CircuitBuilder
from .circuit_build_options import CircuitBuildOptions
from .edge_coloration import edge_coloration


class ZXColorationBuilder(CircuitBuilder):
    name = "zxcoloration"

    def __init__(self, code):
        if code is None:
            raise ValueError("ZXColorationBuilder requires a code instance.")
        self.code = code
        self.build_graph()
        self.color_edges()
        # Expose drawing on the code object for convenience after circuit build.
        self.code.draw_graph = self.draw_graph

    # Draw the Tanner graph of the code.
    def draw_graph(
        self,
        part="all",
        draw_edges=True,
        x_scale=3.0,
        y_scale=3.0,
        node_size=100,
        font_size=8,
        figsize=None,
    ):
        code = self.code
        part = part.lower()

        if part in ("all", "tanner", "full"):
            graph = code.graph
        elif part in ("z", "zcheck", "z-check"):
            graph = code.graph_Z
        elif part in ("x", "xcheck", "x-check"):
            graph = code.graph_X
        else:
            raise ValueError("part must be one of: all, z, x")

        pos = nx.get_node_attributes(code.graph, "pos")
        if x_scale != 1.0 or y_scale != 1.0:
            pos = {k: (v[0] * x_scale, v[1] * y_scale) for k, v in pos.items()}
        nodes = list(graph.nodes())
        node_colors = [code.node_colors[node] for node in nodes]

        if figsize is not None:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=figsize)
            except ImportError:
                pass

        if not draw_edges:
            nx.draw(
                graph,
                pos,
                nodelist=nodes,
                node_color=node_colors,
                node_size=node_size,
                font_size=font_size,
                with_labels=True,
                font_color="white",
            )
            return

        palette = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan", "gold", "navy",
            "teal", "crimson", "darkorange", "slateblue",
            "seagreen", "indigo", "peru", "darkcyan", 
            "firebrick", "darkgreen", "sienna", "dodgerblue",
        ]

        def _edge_color_map(colored_edges):
            mapping = {}
            for color_idx, edges in colored_edges.items():
                for u, v in edges:
                    mapping[frozenset((u, v))] = color_idx
            return mapping

        if part in ("all", "tanner", "full"):
            edge_color_idx = {}
            edge_color_idx.update(_edge_color_map(code.colored_edges_Z))
            edge_color_idx.update(_edge_color_map(code.colored_edges_X))
        elif part in ("z", "zcheck", "z-check"):
            edge_color_idx = _edge_color_map(code.colored_edges_Z)
        else:
            edge_color_idx = _edge_color_map(code.colored_edges_X)

        edges = list(graph.edges())
        edge_colors = [
            palette[edge_color_idx[frozenset((u, v))] % len(palette)]
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
        if code.hz is None or code.hx is None:
            raise ValueError("ZXColorationBuilder requires code.hz and code.hx to be set.")
        if code.hz.shape[1] != code.hx.shape[1]:
            raise ValueError("code.hz and code.hx must have the same number of columns (data qubits).")

        code.graph = nx.Graph()
        code.basis_colors = {"Z": "green", "X": "purple"}

        code.node_colors = []  # 'blue' for data qubits, 'green' for zcheck qubits, 'purple' for xcheck qubits
        code.edges_Z = []  # edges of the Tanner graph for Z checks
        code.edges_X = []  # edges of the Tanner graph for X checks
        code.colored_edges_Z = {}  # key is color, values are edges for Z checks
        code.colored_edges_X = {}  # key is color, values are edges for X checks
        code.num_colors = {"Z": 0, "X": 0}

        n_data = code.hz.shape[1]
        n_z = code.hz.shape[0]
        n_x = code.hx.shape[0]

        data_qubits = np.arange(n_data, dtype=int)
        zcheck_qubits = np.arange(n_data, n_data + n_z, dtype=int)
        xcheck_qubits = np.arange(n_data + n_z, n_data + n_z + n_x, dtype=int)

        for i in range(n_data):
            code.graph.add_node(int(data_qubits[i]), pos=(i, 0))
            code.node_colors += ["blue"]

        for i in range(n_z):
            node = int(zcheck_qubits[i])
            code.graph.add_node(node, pos=(i, -1))
            code.node_colors += ["green"]

        for i in range(n_x):
            node = int(xcheck_qubits[i])
            code.graph.add_node(node, pos=(i, 1))
            code.node_colors += ["purple"]

        code.data_qubits = data_qubits
        code.zcheck_qubits = zcheck_qubits
        code.xcheck_qubits = xcheck_qubits
        code.check_qubits = np.concatenate((zcheck_qubits, xcheck_qubits))
        code.all_qubits = np.arange(n_data + n_z + n_x, dtype=int)

        def _add_edge(basis, control, target):
            edge = (control, target)
            if basis == "Z":
                code.edges_Z += [edge]
            else:
                code.edges_X += [edge]
            code.graph.add_edge(control, target, color=code.basis_colors[basis])

        for z_row, data_col in np.argwhere(code.hz == 1):
            control = int(data_col)
            target = int(n_data + z_row)
            _add_edge("Z", control, target)

        for x_row, data_col in np.argwhere(code.hx == 1):
            control = int(n_data + n_z + x_row)
            target = int(data_col)
            _add_edge("X", control, target)

        code.graph_Z = nx.Graph()
        code.graph_Z.add_nodes_from([int(q) for q in code.data_qubits], bipartite=0)
        code.graph_Z.add_nodes_from([int(q) for q in code.zcheck_qubits], bipartite=1)
        for control, target in code.edges_Z:
            code.graph_Z.add_edge(control, target, orientation=(control, target))

        code.graph_X = nx.Graph()
        code.graph_X.add_nodes_from([int(q) for q in code.xcheck_qubits], bipartite=0)
        code.graph_X.add_nodes_from([int(q) for q in code.data_qubits], bipartite=1)
        for control, target in code.edges_X:
            code.graph_X.add_edge(control, target, orientation=(control, target))

        return code.graph

    def color_edges(self):
        code = self.code
        code.colored_edges_Z = edge_coloration(code.graph_Z)
        code.colored_edges_X = edge_coloration(code.graph_X)
        code.num_colors["Z"] = len(code.colored_edges_Z)
        code.num_colors["X"] = len(code.colored_edges_X)
        # Total number of entangling gate layers across all edge-color groups.
        code.depth = sum(code.num_colors.values())
        return

    def get_coloration_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis='Z',
        circuit_build_options=None,
    ):
        """
        Returns the full Stim circuit for circuit-level simulations using ZX coloration.
        Errors occur at each and every gate.

        :param error_model: ErrorModel that stores idle/single-/two-qubit/spam error parameters.
        :param num_rounds: Number of stabilizer measurement rounds
        :param basis: Basis in which logical codewords are stored. Options are 'Z' and 'X'.
        :param circuit_build_options: CircuitBuildOptions for detector/noisy-round/final-meas behavior.
        :return circuit: Stim circuit
        """
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
        circ = Circuit(code.all_qubits)

        def _flatten_edge_pairs(edge_pairs):
            return [q for control, target in edge_pairs for q in (control, target)]

        def _add_stabilizer_round():
            # Z checks
            circ.add_reset(code.zcheck_qubits)
            circ.add_tick()
            for color in range(code.num_colors['Z']):
                edges = code.colored_edges_Z[color]
                circ.add_cnot_layer(_flatten_edge_pairs(edges))
            circ.add_measure_layer(code.zcheck_qubits)

            # X checks
            circ.add_reset(code.xcheck_qubits)
            circ.add_tick()
            circ.add_hadamard_layer(code.xcheck_qubits)
            for color in range(code.num_colors['X']):
                edges = code.colored_edges_X[color]
                circ.add_cnot_layer(_flatten_edge_pairs(edges))
            circ.add_hadamard_layer(code.xcheck_qubits)
            circ.add_measure_layer(code.xcheck_qubits)
            return

        ################## Logical state prep ##################
        if circuit_build_options.noisy_zeroth_round:
            circ.set_error_model(error_model)
        else:
            circ.set_error_model(ErrorModel.zero())

        circ.add_reset(code.data_qubits, basis)
        _add_stabilizer_round()

        if basis == 'Z':
            for i in range(1, len(code.zcheck_qubits) + 1)[::-1]:
                circ.add_detector([len(code.xcheck_qubits) + i])
        elif basis == 'X':
            for i in range(1, len(code.xcheck_qubits) + 1)[::-1]:
                circ.add_detector([i])

        ############## Logical memory w/ noise ###############
        circ.set_error_model(error_model)

        if num_rounds > 0:
            circ.start_loop(num_rounds)

            _add_stabilizer_round()

            if get_Z_detectors:
                for i in range(1, len(code.zcheck_qubits) + 1)[::-1]:
                    ind = len(code.xcheck_qubits) + i
                    circ.add_detector([ind, ind + len(code.check_qubits)])
            if get_X_detectors:
                for i in range(1, len(code.xcheck_qubits) + 1)[::-1]:
                    circ.add_detector([i, i + len(code.check_qubits)])

            circ.end_loop()

        ################## Logical measurement ##################
        if not circuit_build_options.noisy_final_meas:
            circ.set_error_model(ErrorModel.zero())

        circ.add_measure(code.data_qubits, basis)

        if basis == 'Z':
            for i in range(1, len(code.zcheck_qubits) + 1)[::-1]:
                inds = np.array([len(code.data_qubits) + len(code.xcheck_qubits) + i])
                inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hz[len(code.zcheck_qubits) - i, :] == 1)[0]))
                circ.add_detector(inds)

            for i in range(len(code.lz)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lz[i, :] == 1)[0])

        elif basis == 'X':
            for i in range(1, len(code.xcheck_qubits) + 1)[::-1]:
                inds = np.array([len(code.data_qubits) + i])
                inds = np.concatenate((inds, len(code.data_qubits) - np.where(code.hx[len(code.xcheck_qubits) - i, :] == 1)[0]))
                circ.add_detector(inds)

            for i in range(len(code.lx)):
                circ.add_observable(i, len(code.data_qubits) - np.where(code.lx[i, :] == 1)[0])

        return stim.Circuit(circ.circuit)
