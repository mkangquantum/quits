import networkx as nx
import pytest

from quits.qldpc_code.circuit_construction.edge_coloration import edge_coloration


def _assert_proper_coloration(graph, colored_edges):
    colored_edge_count = sum(len(edges) for edges in colored_edges.values())
    assert colored_edge_count == graph.number_of_edges()

    edge_to_color = {}
    for color, edges in colored_edges.items():
        assert isinstance(color, int)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            u, v = edge
            assert graph.has_edge(u, v)
            assert (u, v) not in edge_to_color
            edge_to_color[(u, v)] = color

    incident_colors = {node: set() for node in graph.nodes()}
    for u, v in graph.edges():
        if (u, v) in edge_to_color:
            color = edge_to_color[(u, v)]
        elif (v, u) in edge_to_color:
            color = edge_to_color[(v, u)]
        else:
            raise AssertionError(f"Missing color for edge {(u, v)}")
        assert color not in incident_colors[u]
        assert color not in incident_colors[v]
        incident_colors[u].add(color)
        incident_colors[v].add(color)



def test_edge_coloration_dense_irregular_bipartite_graph_uses_delta_colors():
    graph = nx.Graph()
    left = [f"u{i}" for i in range(6)]
    right = [f"v{i}" for i in range(7)]
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)

    graph.add_edges_from(
        [
            ("u0", "v0"), ("u0", "v1"), ("u0", "v3"), ("u0", "v5"),
            ("u1", "v1"), ("u1", "v2"), ("u1", "v4"), ("u1", "v6"),
            ("u2", "v0"), ("u2", "v2"), ("u2", "v3"), ("u2", "v6"),
            ("u3", "v1"), ("u3", "v3"), ("u3", "v4"), ("u3", "v5"),
            ("u4", "v0"), ("u4", "v2"), ("u4", "v4"), ("u4", "v5"),
            ("u5", "v2"), ("u5", "v3"), ("u5", "v5"), ("u5", "v6"),
        ]
    )

    colored_edges = edge_coloration(graph)
    _assert_proper_coloration(graph, colored_edges)

    delta = max(dict(graph.degree()).values())
    assert len(colored_edges) == delta



def test_edge_coloration_unbalanced_disconnected_bipartite_graph_uses_delta_colors():
    graph = nx.Graph()

    left = [f"a{i}" for i in range(7)]
    right = [f"b{i}" for i in range(9)]
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)

    graph.add_edges_from(
        [
            ("a0", "b0"), ("a0", "b1"), ("a0", "b2"),
            ("a1", "b0"), ("a1", "b2"), ("a1", "b3"),
            ("a2", "b1"), ("a2", "b3"), ("a2", "b4"),
            ("a3", "b5"), ("a3", "b6"),
            ("a4", "b5"), ("a4", "b7"),
            ("a5", "b6"), ("a5", "b7"),
        ]
    )

    # Keep a6 and b8 isolated to test robustness with disconnected/isolated nodes.
    colored_edges = edge_coloration(graph)
    _assert_proper_coloration(graph, colored_edges)

    delta = max(dict(graph.degree()).values())
    assert len(colored_edges) == delta



def test_edge_coloration_raises_on_non_bipartite_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    with pytest.raises(ValueError, match="bipartite"):
        edge_coloration(graph)


def test_edge_coloration_preserves_orientation_attribute():
    graph = nx.Graph()
    left = ["u0", "u1"]
    right = ["v0", "v1"]
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)

    graph.add_edge("u0", "v0", orientation=("v0", "u0"))
    graph.add_edge("u0", "v1", orientation=("u0", "v1"))
    graph.add_edge("u1", "v0", orientation=("u1", "v0"))
    graph.add_edge("u1", "v1", orientation=("v1", "u1"))

    colored_edges = edge_coloration(graph)
    flattened = [edge for edges in colored_edges.values() for edge in edges]
    assert set(flattened) == {
        ("v0", "u0"),
        ("u0", "v1"),
        ("u1", "v0"),
        ("v1", "u1"),
    }
