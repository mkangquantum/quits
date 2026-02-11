from __future__ import annotations

from typing import Hashable

import networkx as nx


WrappedNode = tuple[str, Hashable] | tuple[str, int]


def edge_coloration(graph: nx.Graph) -> dict[int, list[tuple[Hashable, Hashable]]]:
    """Return an exact edge-coloring of a bipartite graph.

    The returned mapping is ``color -> list of edges`` where each edge is an
    oriented tuple ``(u, v)``. If an input edge carries an ``orientation``
    attribute, that tuple is returned; otherwise, orientation defaults to
    left-to-right according to ``networkx.algorithms.bipartite.color``.

    :param graph: Undirected simple NetworkX graph.
    :return: Mapping from color index to a list of edges assigned that color.
    :raises ValueError: If the graph is directed, a multigraph, or non-bipartite.
    :raises RuntimeError: If regularization/decomposition unexpectedly fails.
    """
    _validate_bipartite_graph(graph)

    if graph.number_of_edges() == 0:
        return {}

    part = nx.algorithms.bipartite.color(graph)
    left_orig = [node for node, color in part.items() if color == 0]
    right_orig = [node for node, color in part.items() if color == 1]
    delta = max(dict(graph.degree()).values())

    regular_graph, left_nodes, right_nodes, original_edges, orientation_by_edge = _build_delta_regular_supergraph(
        graph=graph,
        left_orig=left_orig,
        right_orig=right_orig,
        delta=delta,
        part=part,
    )

    colored_edges: dict[int, list[tuple[Hashable, Hashable]]] = {}
    for color in range(delta):
        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
            regular_graph,
            top_nodes=left_nodes,
        )

        if any(node not in matching for node in left_nodes):
            raise RuntimeError("Expected a perfect matching in delta-regular bipartite graph.")

        matched_edges: list[tuple[WrappedNode, WrappedNode]] = []
        for left_node in left_nodes:
            right_node = matching[left_node]
            matched_edges.append((left_node, right_node))

            if (left_node, right_node) in original_edges:
                key = frozenset((left_node, right_node))
                edge = orientation_by_edge[key]
                if color not in colored_edges:
                    colored_edges[color] = [edge]
                else:
                    colored_edges[color].append(edge)

        regular_graph.remove_edges_from(matched_edges)

    colored_edge_count = sum(len(edges) for edges in colored_edges.values())
    if colored_edge_count != graph.number_of_edges():
        raise RuntimeError("Coloration is incomplete for original graph edges.")

    return colored_edges


def _validate_bipartite_graph(graph: nx.Graph) -> None:
    if graph is None:
        raise ValueError("graph must be a networkx.Graph instance.")
    if graph.is_directed():
        raise ValueError("edge_coloration requires an undirected graph.")
    if graph.is_multigraph():
        raise ValueError("edge_coloration requires a simple graph (no multi-edges).")
    if not nx.is_bipartite(graph):
        raise ValueError("edge_coloration requires a bipartite graph.")


def _build_delta_regular_supergraph(
    graph: nx.Graph,
    left_orig: list[Hashable],
    right_orig: list[Hashable],
    delta: int,
    part: dict[Hashable, int],
) -> tuple[
    nx.Graph,
    list[WrappedNode],
    list[WrappedNode],
    set[tuple[WrappedNode, WrappedNode]],
    dict[frozenset[WrappedNode], tuple[Hashable, Hashable]],
]:
    base_size = max(len(left_orig), len(right_orig), delta)
    max_extra = len(left_orig) + len(right_orig) + delta + 5

    for extra in range(max_extra + 1):
        target_size = base_size + extra
        attempt = _try_build_regular_supergraph(
            graph=graph,
            left_orig=left_orig,
            right_orig=right_orig,
            delta=delta,
            part=part,
            target_size=target_size,
        )
        if attempt is not None:
            return attempt

    raise RuntimeError("Failed to regularize bipartite graph for exact edge coloration.")


def _try_build_regular_supergraph(
    graph: nx.Graph,
    left_orig: list[Hashable],
    right_orig: list[Hashable],
    delta: int,
    part: dict[Hashable, int],
    target_size: int,
) -> tuple[
    nx.Graph,
    list[WrappedNode],
    list[WrappedNode],
    set[tuple[WrappedNode, WrappedNode]],
    dict[frozenset[WrappedNode], tuple[Hashable, Hashable]],
] | None:
    left_nodes: list[WrappedNode] = [("orig_left", node) for node in left_orig]
    right_nodes: list[WrappedNode] = [("orig_right", node) for node in right_orig]

    left_nodes.extend(("dummy_left", i) for i in range(target_size - len(left_nodes)))
    right_nodes.extend(("dummy_right", i) for i in range(target_size - len(right_nodes)))

    left_deficit: dict[WrappedNode, int] = {}
    right_deficit: dict[WrappedNode, int] = {}

    for node in left_nodes:
        if node[0] == "orig_left":
            degree = graph.degree(node[1])
        else:
            degree = 0
        left_deficit[node] = delta - degree

    for node in right_nodes:
        if node[0] == "orig_right":
            degree = graph.degree(node[1])
        else:
            degree = 0
        right_deficit[node] = delta - degree

    if any(v < 0 for v in left_deficit.values()) or any(v < 0 for v in right_deficit.values()):
        return None

    total_left = sum(left_deficit.values())
    total_right = sum(right_deficit.values())
    if total_left != total_right:
        return None

    original_edges: set[tuple[WrappedNode, WrappedNode]] = set()
    orientation_by_edge: dict[frozenset[WrappedNode], tuple[Hashable, Hashable]] = {}
    for u, v, data in graph.edges(data=True):
        if part[u] == 0:
            left_node = ("orig_left", u)
            right_node = ("orig_right", v)
        else:
            left_node = ("orig_left", v)
            right_node = ("orig_right", u)
        original_edges.add((left_node, right_node))
        orientation = data.get("orientation")
        if orientation is not None:
            if not isinstance(orientation, tuple) or len(orientation) != 2:
                raise ValueError("edge 'orientation' attribute must be a 2-tuple.")
            oriented_edge = (orientation[0], orientation[1])
        else:
            oriented_edge = (left_node[1], right_node[1])
        orientation_by_edge[frozenset((left_node, right_node))] = oriented_edge

    flow_graph = nx.DiGraph()
    source = "source"
    sink = "sink"

    for node in left_nodes:
        flow_graph.add_edge(source, ("L", node), capacity=left_deficit[node])
    for node in right_nodes:
        flow_graph.add_edge(("R", node), sink, capacity=right_deficit[node])

    for left_node in left_nodes:
        for right_node in right_nodes:
            if (left_node, right_node) not in original_edges:
                flow_graph.add_edge(("L", left_node), ("R", right_node), capacity=1)

    flow_value, flow = nx.maximum_flow(flow_graph, source, sink)
    if flow_value != total_left:
        return None

    regular_graph = nx.Graph()
    regular_graph.add_nodes_from(left_nodes, bipartite=0)
    regular_graph.add_nodes_from(right_nodes, bipartite=1)
    regular_graph.add_edges_from(original_edges)

    for left_node in left_nodes:
        row = flow[("L", left_node)]
        for right_key, value in row.items():
            if value <= 0 or not isinstance(right_key, tuple) or len(right_key) != 2 or right_key[0] != "R":
                continue
            right_node = right_key[1]
            regular_graph.add_edge(left_node, right_node)

    if any(regular_graph.degree(node) != delta for node in left_nodes + right_nodes):
        return None

    return regular_graph, left_nodes, right_nodes, original_edges, orientation_by_edge


__all__ = ["edge_coloration"]
