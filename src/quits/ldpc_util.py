"""
@author: Hanwen Yao, Mert Gökduman
LDPC utilities focused on:
  - Generating classical LDPC parity-check matrices
  - Computing / optimizing girth of the Tanner graph
"""

import numpy as np
import random
from collections import deque
from typing import Optional, Tuple
from .gf2_util import gf2_rank


# ============================================================
# LDPC generation (configuration model; may include multi-edges)
# ============================================================

def generate_ldpc(n: int, dv: int, dc: int) -> np.ndarray:
    """
    Generate an (m x n) LDPC parity-check matrix using a configuration model.
    Note: entries may exceed 1 (multi-edges).
    """
    if (n * dv) % dc != 0:
        raise ValueError("n * dv must be divisible by dc")
    m = (n * dv) // dc  # number of rows (check nodes)

    # Create a list with dv copies of each column index.
    col_sockets = []
    for col in range(n):
        col_sockets.extend([col] * dv)

    # Create a list with dc copies of each row index.
    row_sockets = []
    for row in range(m):
        row_sockets.extend([row] * dc)

    # There are the same total number of sockets in each list.
    assert len(col_sockets) == len(row_sockets)

    # Shuffle the row sockets and pair them with column sockets.
    random.shuffle(row_sockets)
    H = np.zeros((m, n), dtype=int)
    for col, row in zip(col_sockets, row_sockets):
        H[row, col] += 1    # Increment the entry to count multiple edges 
    return H


def has_duplicate_edges(H: np.ndarray) -> bool:
    """
    Returns True if the binary matrix H has any entry greater than 1,
    indicating the presence of multiple edges.
    """
    return bool(np.any(H > 1))


# ============================================================
# Girth computation on Tanner graph (binary support of H)
# ============================================================

def build_bipartite_adjacency(H: np.ndarray):
    """
    Build adjacency list for the bipartite graph represented by parity-check matrix H.
    Rows -> check nodes, columns -> bit nodes.
    """
    H = (np.asarray(H) > 0).astype(np.uint8)
    M, N = H.shape
    adjacency_list = [[] for _ in range(M + N)]
    for i in range(M):
        cols = np.where(H[i, :] == 1)[0]
        for j in cols:
            # Add an edge i <-> (M + j)
            adjacency_list[i].append(M + j)
            adjacency_list[M + j].append(i)
    return adjacency_list


def bfs_shortest_cycle(adjacency_list, start: int) -> int:
    """
    Performs BFS from 'start' node to find the shortest cycle reachable
    from this node. Returns the length of that cycle or infinity if none found.
    """
    dist = [-1] * len(adjacency_list)   # distance array
    dist[start] = 0
    q = deque([start])
    min_cycle_len = float("inf")

    while q:
        cur = q.popleft()
        for nb in adjacency_list[cur]:
            if dist[nb] == -1:
                # If neighbor is unvisited, set distance and enqueue
                dist[nb] = dist[cur] + 1
                q.append(nb)
            else:
                # If neighbor is visited (dist[neighbor] != -1)
                # and it's not the immediate parent of current in BFS tree,
                # then we have found a cycle.
                #
                # In an undirected BFS, the immediate parent of 'current' 
                # is the node with dist[current] - 1, i.e., one less distance.
                # A typical check is: if dist[neighbor] >= dist[current], 
                # we found a cycle that is not just an edge back to the parent.
                if dist[nb] >= dist[cur]:
                    cycle_len = dist[nb] + dist[cur] + 1
                    if cycle_len < min_cycle_len:
                        min_cycle_len = cycle_len

    return min_cycle_len


def girth_of_bipartite(adjacency_list) -> int:
    """
    Computes the girth (length of shortest cycle) of the bipartite graph
    represented by 'adjacency_list'.
    Returns float('inf') if there is no cycle.
    """
    best = float("inf")
    for node in range(len(adjacency_list)):
        best = min(best, bfs_shortest_cycle(adjacency_list, node))
    return best


def compute_girth_from_parity_check(H: np.ndarray) -> int:
    """
    Given a parity-check matrix H (M x N),
    build its bipartite adjacency list, then compute and return the girth.
    """
    return girth_of_bipartite(build_bipartite_adjacency(H))


# ============================================================
# Girth-oriented optimization (edge swapping on multi-edge Tanner)
# ============================================================

def get_neighbors(node, H):
    """
    Given a node in the Tanner graph (represented as a tuple):
      - ('v', i) for a variable node i, or
      - ('c', j) for a check node j,
    return a list of (neighbor, mult) pairs.
    Here, mult is the multiplicity (i.e. the count from H).
    """
    neighbors = []
    if node[0] == "v":
        i = node[1]
        # Neighbors are check nodes j with H[j, i] > 0.
        for j in range(H.shape[0]):
            if H[j, i] > 0:
                neighbors.append((("c", j), H[j, i]))
    else:
        j = node[1]
        # Neighbors are variable nodes i with H[j, i] > 0.
        for i in range(H.shape[1]):
            if H[j, i] > 0:
                neighbors.append((("v", i), H[j, i]))
    return neighbors


def dfs_count_paths(current, target, H, v_exclude, visited, depth, max_depth):
    """
    Recursively count the number of simple paths from 'current' to 'target'
    in the Tanner graph (given by H) with depth at most max_depth.
    We multiply counts by the edge multiplicities.
    
    v_exclude is the index of the variable node (v) for which we are counting cycles.
    We do not allow v_exclude as an intermediate node (except when it is the target).
    
    visited is a set of nodes (tuples) already on the current path (to avoid revisiting).
    
    Returns:
       A dictionary mapping a path length (an integer) to the number of paths
       (with multiplicative weight) that reach the target with that exact length.
       If no path is found, returns an empty dictionary.
    """    
    if depth > max_depth:
        return {}
    if current == target:
        # We require a nonzero length path.
        return {depth: 1} if depth > 0 else {}

    counts = {}
    for neighbor, mult in get_neighbors(current, H):
        # Skip v_exclude if it is not the target (to ensure v appears only at start/end)
        if neighbor[0] == "v" and neighbor[1] == v_exclude and neighbor != target:
            continue
        # Enforce simplicity: do not revisit nodes (except the target)
        if neighbor != target and neighbor in visited:
            continue
        # Mark neighbor as visited (if not target)
        if neighbor != target:
            visited.add(neighbor)
        sub = dfs_count_paths(neighbor, target, H, v_exclude, visited, depth + 1, max_depth)
        if neighbor != target:
            visited.remove(neighbor)
        # Multiply the counts by the multiplicity of the edge used
        for d, cnt in sub.items():
            counts[d] = counts.get(d, 0) + cnt * mult
    return counts


def shortest_cycle_and_count_for_variable(H, v, max_depth=10):
    """
    For a given parity-check matrix H (possibly with duplicate edges) and a given variable node v,
    compute:
      - lv: the length of the shortest cycle in the Tanner graph that involves v, and
      - mv: the number of cycles (counted with multiplicity) of length lv that involve v.
    
    Here, a cycle is defined as a simple cycle in the bipartite Tanner graph.
    (Cycles of length 2 arise when a check node is incident on v more than once.)
    
    For cycles of length >= 4, we proceed as follows:
      For each edge (v, c), we remove one copy of that edge and count the number
      of simple paths from ('c', c) back to ('v', v) using a DFS (with depth limit max_depth).
      Each such path of length d gives a cycle of length d + 1 (restoring the removed edge).
    
    Since any such cycle touches v by exactly two edges, each cycle is found twice overall.
    
    Returns:
       (lv, mv), where lv is the shortest cycle length (or None if no cycle is found)
       and mv is the number of cycles of that length.
    """
    m = H.shape[0]
    check_neighbors = [c for c in range(m) if H[c, v] > 0]

    best = float("inf")
    ways_sum = 0

    # 2-cycles (multi-edge)
    for c in check_neighbors:
        if H[c, v] > 1:
            # A 2-cycle exists from v -> c -> v.
            if 2 < best:
                best = 2
                ways_sum = 0
            if 2 == best:
                # There are choose(H[c,v], 2) cycles contributed by check node c.
                ways_sum += (H[c, v] * (H[c, v] - 1)) // 2
    if best == 2:
        return 2, ways_sum

    # cycles >= 4
    for c in check_neighbors:
        H[c, v] -= 1    # Remove one copy of edge (v, c)
        # Start DFS from node ('c', c) with target ('v', v).
        # We begin with visited containing the start.
        visited = {("c", c)}
        result = dfs_count_paths(("c", c), ("v", v), H, v_exclude=v, visited=visited, depth=0, max_depth=max_depth)
        H[c, v] += 1    # Restore the edge.
        if result:
            dmin = min(result.keys())
            cycle_len = dmin + 1    # add back the removed edge
            if cycle_len < best:
                best = cycle_len
                ways_sum = result[dmin]
            elif cycle_len == best:
                ways_sum += result[dmin]

    if best == float("inf"):
        return None, 0
    # Each cycle of length >= 4 is counted twice (once for each edge incident on v).
    return best, ways_sum // 2


def _score_key(score):
    """
    Given a score (l, m), return a key for lexicographic comparison.
    We want (l1, m1) < (l2, m2) if l1 < l2 or (l1 == l2 and m1 > m2).
    Mapping (l, m) -> (l, -m) achieves this.
    """    
    l, m = score
    return (l, -m)


def _is_better(new1, new2, old1, old2):
    """
    Compare two pairs of scores for variable nodes v1 and v2.
    Each score is a tuple (l, m).
    
    We first compute:
      new_min = min(new_score_v1, new_score_v2)  (using our custom ordering)
      old_min = min(old_score_v1, old_score_v2)
    
    If new_min is better (i.e. smaller in our order) than old_min, we return True.
    Otherwise, if they are equal, we compare the corresponding maximums and return True
    if the new maximum is better than the old maximum.
    
    Otherwise, return False.
    """    
    new_min = min(new1, new2, key=_score_key)
    old_min = min(old1, old2, key=_score_key)
    if _score_key(new_min) > _score_key(old_min):
        return True
    if _score_key(new_min) == _score_key(old_min):
        new_max = max(new1, new2, key=_score_key)
        old_max = max(old1, old2, key=_score_key)
        return _score_key(new_max) > _score_key(old_max)
    return False


def enumerate_edges(H):
    """
    Return a list of edge instances from the Tanner graph.
    Each edge is represented as a tuple (v, c) where v is the variable node (column index)
    and c is the check node (row index). Duplicate edges appear as multiple entries.
    """    
    m, n = H.shape
    edges = []
    for c in range(m):
        for v in range(n):
            for _ in range(H[c, v]):     # if H[c,v] > 0, add that many copies
                edges.append((v, c))
    return edges


def optimize_ldpc(H: np.ndarray, rounds: int, max_depth: int = 10, verbose: bool = False) -> np.ndarray:
    """
    Given a parity-check matrix H (which may contain duplicate edges),
    perform a number of rounds of random edge shuffles.
    
    For each round:
      - Randomly select two edge instances (v1, c1) and (v2, c2) from the Tanner graph.
      - Compute the current scores (lv, mv) for variable nodes v1 and v2.
      - Replace the edges (v1, c1) and (v2, c2) by (v1, c2) and (v2, c1):
            * Decrease H[c1, v1] and H[c2, v2] by 1.
            * Increase H[c1, v2] and H[c2, v1] by 1.
      - Recompute the scores for v1 and v2.
      - If the new pair of scores is "better" (as determined by is_better), keep the change;
        otherwise, revert the swap.
    
    Returns the modified matrix H.
    """
    H = np.asarray(H, dtype=int, order="C")
    for _ in range(rounds):
        # Enumerate all edge instances.
        edges = enumerate_edges(H)
        if len(edges) < 2:
            break   # Not enough edges to swap.

        # Randomly pick two distinct edge instances.
        (v1, c1), (v2, c2) = random.sample(edges, 2)

        # Save the old scores for v1 and v2.
        old1 = shortest_cycle_and_count_for_variable(H, v1, max_depth)
        old2 = shortest_cycle_and_count_for_variable(H, v2, max_depth)

        # Perform the swap: remove one copy of (v1, c1) and (v2, c2)
        # and add one copy of (v1, c2) and (v2, c1).
        H[c1, v1] -= 1
        H[c2, v2] -= 1
        H[c1, v2] += 1
        H[c2, v1] += 1

        # Recompute the scores for v1 and v2 after the swap.
        new1 = shortest_cycle_and_count_for_variable(H, v1, max_depth)
        new2 = shortest_cycle_and_count_for_variable(H, v2, max_depth)

        # Check the condition: if the new scores are "better" keep the swap, else revert.
        if _is_better(new1, new2, old1, old2):
            # Keep the swap (optionally, print or log the improvement).
            if verbose:
                print(f"{old1},{old2} -> {new1},{new2}")
        else:
            # Revert the swap.
            H[c1, v1] += 1
            H[c2, v2] += 1
            H[c1, v2] -= 1
            H[c2, v1] -= 1
    return H


# ============================================================
# High-level helper: generate an LDPC with "good girth"
# ============================================================

def generate_ldpc_good_girth(
    n: int,
    dv: int,
    dc: int,
    target_girth: int,
    max_outer_iters: int = 10,
    rounds_per_iter: int = 100,
    max_depth: int = 10,
    require_full_row_rank: bool = True,
    require_no_multi_edges: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate & optimize an LDPC parity-check matrix until girth >= target_girth
    and (optionally) other constraints are satisfied.

    Returns: (H, girth)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    H = generate_ldpc(n, dv, dc)

    for iter in range(max_outer_iters):
        H = optimize_ldpc(H, rounds=rounds_per_iter, max_depth=max_depth, verbose=True)
        g = compute_girth_from_parity_check(H)
        print('Iteration {}/{}: Girth = {}'.format(iter + 1, max_outer_iters, g))

        if require_no_multi_edges and has_duplicate_edges(H):
            continue
        if require_full_row_rank and gf2_rank(H % 2) < H.shape[0]:
            continue
        if g >= target_girth:
            return H, g

    # return best-effort
    return H % 2, compute_girth_from_parity_check(H)
