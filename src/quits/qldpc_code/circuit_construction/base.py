from dataclasses import dataclass
from numbers import Integral

import networkx as nx


@dataclass(frozen=True)
class EdgeLayering:
    layers: list[list[tuple[int, int]]]


class CircuitBuilder:
    name = None

    def build(self, code, **opts):
        raise NotImplementedError

    def draw_graph(
        self,
        layout=None,
        part="all",   # "node", "all", i (0 <= i <= code.depth-1)
        draw_edges=True,
        x_scale=3.0,
        y_scale=3.0,
        center_checks=True,
        curved_edges=False,
        node_size=100,
        font_size=8,
        figsize=None,
    ):
        graph = self._get_graph_for_draw(part)
        part = self._normalize_draw_part(part)
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

        if part != "node" and draw_edges:
            layering = self._validate_edge_layering(graph, self._get_edge_layering(graph))
            edge_batches = self._get_edge_draw_batches(graph, part, layering, curved_edges=curved_edges)
            for batch in edge_batches:
                edge_kwargs = {
                    "edgelist": batch["edgelist"],
                    "edge_color": batch["edge_color"],
                }
                if batch["connectionstyle"] is not None:
                    # Force FancyArrowPatch rendering so connectionstyle is honored
                    # even for undirected graphs, while keeping arrowheads hidden.
                    edge_kwargs.update(
                        {
                            "arrows": True,
                            "arrowstyle": "-",
                            "connectionstyle": batch["connectionstyle"],
                        }
                    )
                nx.draw_networkx_edges(graph, pos, **edge_kwargs)

        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=font_size,
            font_color="white",
        )

        try:
            import matplotlib.pyplot as plt

            plt.gca().set_axis_off()
        except ImportError:
            pass

    def _normalize_draw_part(self, part):
        if isinstance(part, bool):
            raise ValueError(
                f"For {type(self).__name__} draw_graph, part must be 'node', 'all', "
                "or a non-negative integer layer index."
            )
        if isinstance(part, Integral):
            return int(part)
        if isinstance(part, str):
            normalized = part.strip().lower()
            if normalized in ("node", "all"):
                return normalized
            if normalized.isdigit():
                return int(normalized)
        raise ValueError(
            f"For {type(self).__name__} draw_graph, part must be 'node', 'all', "
            "or a non-negative integer layer index."
        )

    def _get_graph_for_draw(self, part):
        code = getattr(self, "code", None)
        if code is None or not hasattr(code, "graph"):
            raise ValueError(f"{type(self).__name__}.draw_graph requires an initialized code graph.")
        return code.graph

    def _resolve_positions(self, graph, layout=None, **kwargs):
        raise NotImplementedError

    def _get_node_colors(self, graph):
        raise NotImplementedError

    def _get_edge_layering(self, graph):
        raise NotImplementedError

    def _get_edge_palette(self):
        return [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan", "gold", "navy",
            "teal", "crimson", "darkorange", "slateblue",
            "seagreen", "indigo", "peru", "darkcyan",
            "firebrick", "darkgreen", "sienna", "dodgerblue",
        ]

    def _validate_edge_layering(self, graph, layering):
        if not isinstance(layering, EdgeLayering):
            raise TypeError(f"{type(self).__name__}._get_edge_layering must return EdgeLayering.")
        code = getattr(self, "code", None)
        if code is not None and getattr(code, "depth", None) is not None and len(layering.layers) != code.depth:
            raise ValueError(
                f"{type(self).__name__} edge layering has {len(layering.layers)} layers, "
                f"but code.depth is {code.depth}."
            )

        seen_edges = {}
        for layer_index, edges in enumerate(layering.layers):
            for u, v in edges:
                edge_key = frozenset((u, v))
                prior_layer = seen_edges.get(edge_key)
                if prior_layer is not None:
                    raise ValueError(
                        f"{type(self).__name__} edge {tuple(edge_key)} appears in both layer "
                        f"{prior_layer} and layer {layer_index}."
                    )
                seen_edges[edge_key] = layer_index
        return layering

    def _get_layer_curvature(self, layer_index, num_layers):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_layers == 1:
            return 0.12

        max_abs_rad = 0.18
        min_abs_rad = 0.05
        raw = -max_abs_rad + (2 * max_abs_rad * layer_index / (num_layers - 1))
        if abs(raw) < min_abs_rad:
            raw = min_abs_rad if raw >= 0 else -min_abs_rad
        return raw

    def _get_edge_draw_batches(self, graph, part, layering, *, curved_edges):
        palette = self._get_edge_palette()
        num_layers = len(layering.layers)

        def _connectionstyle(layer_index):
            if not curved_edges:
                return None
            rad = self._get_layer_curvature(layer_index, num_layers)
            return f"arc3,rad={rad}"

        if part == "all":
            edge_layer_index = {}
            for layer_index, edges in enumerate(layering.layers):
                for u, v in edges:
                    edge_layer_index[frozenset((u, v))] = layer_index

            batches = []
            for layer_index, edges in enumerate(layering.layers):
                batches.append(
                    {
                        "edgelist": list(edges),
                        "edge_color": palette[layer_index % len(palette)],
                        "connectionstyle": _connectionstyle(layer_index),
                    }
                )

            unlayered_edges = [
                (u, v)
                for u, v in graph.edges()
                if frozenset((u, v)) not in edge_layer_index
            ]
            if unlayered_edges:
                batches.append(
                    {
                        "edgelist": unlayered_edges,
                        "edge_color": "tab:gray",
                        "connectionstyle": None,
                    }
                )
            return batches

        if not layering.layers:
            raise ValueError(f"For {type(self).__name__} draw_graph, there are no edge layers to draw.")
        if part < 0 or part >= len(layering.layers):
            raise ValueError(
                f"For {type(self).__name__} draw_graph, layer index {part} is out of range; "
                f"expected 0 <= part < {len(layering.layers)}."
            )

        selected_edges = list(layering.layers[part])
        return [
            {
                "edgelist": selected_edges,
                "edge_color": palette[part % len(palette)],
                "connectionstyle": _connectionstyle(part),
            }
        ]
