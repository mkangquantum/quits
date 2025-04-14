"""
@author: Hanwen Yao, Mert Gökduman
"""

import numpy as np
import random

"""
----------------------------------------
Note:

The `generate_ldpc` function constructs a Tanner graph using a simple random permutation approach. This can lead to multiple edges between the same variable and check nodes—i.e., entries in the parity-check matrix `H` may exceed 1. This method corresponds to the configuration model, as described in Antoine Grospellier's thesis.

Once a random `H` is generated, you can call `optimize_ldpc` to improve the graph's girth via edge switching. The `rounds` parameter controls how many edge-switching attempts are made. A switch is accepted only if it improves a scoring function. You can set `rounds` as high as you'd like—it just depends on how much time you're willing to spend. (In fact, you can let it run indefinitely if you're feeling patient.)

Be warned: generating a high-quality code for even moderately large sizes (e.g., `n > 10`) can take a long time. The optimization relies on random edge-switch attempts, which aren't particularly smart.

Check the following example usage.
----------------------------------------
Example Usage:
This script demonstrates how to generate and optimize an LDPC parity-check matrix.
For demonstration, we use small parameters.

# Define LDPC parameters
n = 10        # Number of variable nodes
dv = 3        # Variable node degree
dc = 6        # Check node degree (m = n * dv / dc should be an integer)

# Generate an initial parity-check matrix
H = generate_ldpc(n, dv, dc)
print("Original parity-check matrix H:")
print(H)

# Perform initial optimization
rounds = 100
print(f"\nOptimizing for {rounds} rounds...")
H = optimize_ldpc(H, rounds, max_depth=10)

# Continue optimizing until constraints are satisfied
while has_duplicate_edges(H) or binary_matrix_rank(H) < H.shape[0]:
    
    # Additional optimization rounds
    rounds = 10
    print(f"\nOptimizing for another {rounds} rounds...")
    H = optimize_ldpc(H, rounds, max_depth=10)

# Display the final optimized parity-check matrix
print(f"\nOptimized parity-check matrix H (rank {binary_matrix_rank(H)}):")
print(H)

# Further optimization can be performed if needed.
"""

# Generate an LDPC parity-check matrix H (which may contain duplicate edges)
def generate_ldpc(n, dv, dc):
    # Check that the parameters are consistent.
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
    assert len(col_sockets) == len(row_sockets), "Total number of sockets must match."
    
    # Shuffle the row sockets and pair them with column sockets.
    random.shuffle(row_sockets)
    H = np.zeros((m, n), dtype=int)
    for col, row in zip(col_sockets, row_sockets):
        H[row, col] += 1  # Increment the entry to count multiple edges    
    
    return H

# The following are optimizer functions
def get_neighbors(node, H):
    """
    Given a node in the Tanner graph (represented as a tuple):
      - ('v', i) for a variable node i, or
      - ('c', j) for a check node j,
    return a list of (neighbor, mult) pairs.
    Here, mult is the multiplicity (i.e. the count from H).
    """
    neighbors = []
    if node[0] == 'v':
        i = node[1]
        # Neighbors are check nodes j with H[j, i] > 0.
        for j in range(H.shape[0]):
            if H[j, i] > 0:
                neighbors.append((('c', j), H[j, i]))
    else:
        j = node[1]
        # Neighbors are variable nodes i with H[j, i] > 0.
        for i in range(H.shape[1]):
            if H[j, i] > 0:
                neighbors.append((('v', i), H[j, i]))
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
        if neighbor[0] == 'v' and neighbor[1] == v_exclude and neighbor != target:
            continue
        # Enforce simplicity: do not revisit nodes (except the target)
        if neighbor != target and neighbor in visited:
            continue
        # Mark neighbor as visited (if not target)
        if neighbor != target:
            visited.add(neighbor)
        sub_counts = dfs_count_paths(neighbor, target, H, v_exclude, visited, depth + 1, max_depth)
        if neighbor != target:
            visited.remove(neighbor)
        # Multiply the counts by the multiplicity of the edge used
        for d, cnt in sub_counts.items():
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
    # First, check for 2-cycles.
    check_neighbors = [c for c in range(m) if H[c, v] > 0]
    best = float('inf')
    ways_sum = 0
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

    # Now, search for cycles of length >= 4.
    for c in check_neighbors:
        if H[c, v] > 0:
            # Remove one copy of edge (v, c)
            H[c, v] -= 1
            # Start DFS from node ('c', c) with target ('v', v).
            # We begin with visited containing the start.
            visited = {('c', c)}
            result = dfs_count_paths(('c', c), ('v', v), H, v_exclude=v, visited=visited, depth=0, max_depth=max_depth)
            # Restore the edge.
            H[c, v] += 1
            if result:
                d_min = min(result.keys())
                cycle_len = d_min + 1  # add back the removed edge
                if cycle_len < best:
                    best = cycle_len
                    ways_sum = result[d_min]
                elif cycle_len == best:
                    ways_sum += result[d_min]
    if best == float('inf'):
        return None, 0
    # Each cycle of length >= 4 is counted twice (once for each edge incident on v).
    return best, ways_sum // 2

def score_key(score):
    """
    Given a score (l, m), return a key for lexicographic comparison.
    We want (l1, m1) < (l2, m2) if l1 < l2 or (l1 == l2 and m1 > m2).
    Mapping (l, m) -> (l, -m) achieves this.
    """
    l, m = score
    return (l, -m)

def is_better(new_score_v1, new_score_v2, old_score_v1, old_score_v2):
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
    new_min = min(new_score_v1, new_score_v2, key=score_key)
    old_min = min(old_score_v1, old_score_v2, key=score_key)
    
    if score_key(new_min) > score_key(old_min):
        return True
    elif score_key(new_min) == score_key(old_min):
        new_max = max(new_score_v1, new_score_v2, key=score_key)
        old_max = max(old_score_v1, old_score_v2, key=score_key)
        if score_key(new_max) > score_key(old_max):
            return True
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
            for _ in range(H[c, v]):  # if H[c,v] > 0, add that many copies
                edges.append((v, c))
    return edges

def optimize_ldpc(H, rounds, max_depth=10):
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
    m, n = H.shape
    for _ in range(rounds):
        # Enumerate all edge instances.
        edges = enumerate_edges(H)
        if len(edges) < 2:
            break  # Not enough edges to swap.
        
        # Randomly pick two distinct edge instances.
        (v1, c1), (v2, c2) = random.sample(edges, 2)
        
        # Save the old scores for v1 and v2.
        old_score_v1 = shortest_cycle_and_count_for_variable(H, v1, max_depth)
        old_score_v2 = shortest_cycle_and_count_for_variable(H, v2, max_depth)
        
        # Perform the swap: remove one copy of (v1, c1) and (v2, c2)
        # and add one copy of (v1, c2) and (v2, c1).
        H[c1, v1] -= 1
        H[c2, v2] -= 1
        H[c1, v2] += 1
        H[c2, v1] += 1
        
        # Recompute the scores for v1 and v2 after the swap.
        new_score_v1 = shortest_cycle_and_count_for_variable(H, v1, max_depth)
        new_score_v2 = shortest_cycle_and_count_for_variable(H, v2, max_depth)
        
        # Check the condition: if the new scores are "better" keep the swap, else revert.
        if is_better(new_score_v1, new_score_v2, old_score_v1, old_score_v2):
            # Keep the swap (optionally, print or log the improvement).
            print(f"{old_score_v1},{old_score_v2} -> {new_score_v1},{new_score_v2}")
            pass
        else:
            # Revert the swap.
            H[c1, v1] += 1
            H[c2, v2] += 1
            H[c1, v2] -= 1
            H[c2, v1] -= 1
    return H

# The following are check functions
def has_duplicate_edges(H):
    """
    Returns True if the binary matrix H has any entry greater than 1,
    indicating the presence of multiple edges.
    """
    return np.any(H > 1)

def binary_matrix_rank(A):
    """
    Compute the rank of a binary matrix A over GF(2).

    Parameters:
        A (np.ndarray): a 2D numpy array with binary entries (0 or 1).
        
    Returns:
        rank (int): the rank of A over GF(2).
    """
    # Make a copy of A and reduce modulo 2
    A = A.copy() % 2
    rows, cols = A.shape
    rank = 0
    pivot_row = 0

    for col in range(cols):
        # Find a pivot in column `col` at or below pivot_row
        pivot = None
        for r in range(pivot_row, rows):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            # No pivot in this column, move to next column.
            continue

        # Swap pivot row into position
        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        
        # Eliminate all other 1's in this column (across all rows).
        for r in range(rows):
            if r != pivot_row and A[r, col] == 1:
                # In GF(2), subtraction and addition are the same: use XOR.
                A[r, :] = (A[r, :] + A[pivot_row, :]) % 2

        rank += 1
        pivot_row += 1
        if pivot_row == rows:
            break

    return rank


from ldpc.mod2.mod2_numpy import row_echelon, nullspace, row_basis, row_span
from collections import deque 

# This function is a verbatim implementation from the bposd GitHub library by Joschka Roffe.
# Source: https://github.com/quantumgizmos/bp_osd
def compute_lz(hx, hz):
    """
    Computes the logical Z operators for a CSS code.

    Parameters
    ----------
    hx : numpy.ndarray
        The parity check matrix for X errors.
    hz : numpy.ndarray
        The parity check matrix for Z errors.

    Returns
    -------
    log_ops : numpy.ndarray
        A matrix where each row is a logical Z operator.
    """
    # Compute the kernel (nullspace) of hx: vectors v such that hx @ v = 0
    ker_hx = nullspace(hx)

    # Compute the image (row basis) of hz^T: vectors that can be expressed as hz.T @ w
    im_hzT = row_basis(hz)

    # Stack the image and kernel matrices vertically
    log_stack = np.vstack([im_hzT, ker_hx])

    # Convert to row echelon form to identify dependencies
    row_ech, rank, _, pivot_cols = row_echelon(log_stack.T)

    # Identify which vectors in the kernel are not in the image
    # They correspond to pivot columns beyond the image's rank
    image_rank = im_hzT.shape[0]
    log_op_indices = [i for i in range(image_rank, log_stack.shape[0]) if i in pivot_cols]
    log_ops = log_stack[log_op_indices]

    return log_ops

# This function is a verbatim implementation from the bposd GitHub library by Joschka Roffe.
# Source: https://github.com/quantumgizmos/bp_osd
def compute_lz_and_lx(hx, hz):
    """
    Computes both logical Z and logical X operators for a CSS code.

    Parameters
    ----------
    hx : numpy.ndarray
        The parity check matrix for X errors.
    hz : numpy.ndarray
        The parity check matrix for Z errors.

    Returns
    -------
    lz : numpy.ndarray
        A matrix where each row is a logical Z operator.
    lx : numpy.ndarray
        A matrix where each row is a logical X operator.
    """
    # Compute Logical Z operators using (hx, hz)
    lz = compute_lz(hx, hz)

    # Compute Logical X operators using (hz, hx)
    lx = compute_lz(hz, hx)

    return lz, lx

# This function is a verbatim implementation from the bposd GitHub library by Joschka Roffe.
# Source: https://github.com/quantumgizmos/bp_osd
def compute_code_distance(H):
    '''
    Computes the distance of the code given by parity check matrix H. The code distance is given by the minimum weight of a nonzero codeword.

    Note
    ----
    The runtime of this function scales exponentially with the block size. In practice, computing the code distance of codes with block lengths greater than ~10 will be very slow.

    Parameters
    ----------
    H: numpy.ndarray
        The parity check matrix
    
    Returns
    -------
    int
        The code distance
    '''
    ker=nullspace(H)

    if len(ker)==0: return np.inf #return np.inf if the kernel is empty (eg. infinite code distance)

    cw=row_span(ker) #nonzero codewords

    return np.min(np.sum(cw, axis=1))

# ======================================================================
# The following functions written by Mert are for girth computation 
# ======================================================================
def build_bipartite_adjacency(H: np.ndarray):
    """
    Build adjacency list for the bipartite graph represented by parity-check matrix H.
    Rows -> check nodes, columns -> bit nodes.
    """
    M, N = H.shape
    # Total number of nodes = M (checks) + N (bits)
    adjacency_list = [[] for _ in range(M + N)]
    
    for i in range(M):       # for each check node
        for j in range(N):   # for each bit node
            if H[i, j] == 1:
                # Add an edge i <-> (M + j)
                adjacency_list[i].append(M + j)
                adjacency_list[M + j].append(i)
    
    return adjacency_list

def bfs_shortest_cycle(adjacency_list, start):
    """
    Performs BFS from 'start' node to find the shortest cycle reachable
    from this node. Returns the length of that cycle or infinity if none found.
    """
    dist = [-1] * len(adjacency_list)  # distance array
    dist[start] = 0
    q = deque([start])
    
    min_cycle_len = float('inf')
    
    while q:
        current = q.popleft()
        
        for neighbor in adjacency_list[current]:
            if dist[neighbor] == -1:
                # If neighbor is unvisited, set distance and enqueue
                dist[neighbor] = dist[current] + 1
                q.append(neighbor)
            else:
                # If neighbor is visited (dist[neighbor] != -1)
                # and it's not the immediate parent of current in BFS tree,
                # then we have found a cycle.
                #
                # In an undirected BFS, the immediate parent of 'current' 
                # is the node with dist[current] - 1, i.e., one less distance.
                # A typical check is: if dist[neighbor] >= dist[current], 
                # we found a cycle that is not just an edge back to the parent.
                
                if dist[neighbor] >= dist[current]:
                    cycle_length = dist[neighbor] + dist[current] + 1
                    min_cycle_len = min(min_cycle_len, cycle_length)
    
    return min_cycle_len

def girth_of_bipartite(adjacency_list):
    """
    Computes the girth (length of shortest cycle) of the bipartite graph
    represented by 'adjacency_list'.
    Returns float('inf') if there is no cycle.
    """
    min_cycle = float('inf')
    for node in range(len(adjacency_list)):
        cycle_len = bfs_shortest_cycle(adjacency_list, node)
        min_cycle = min(min_cycle, cycle_len)
    return min_cycle

def compute_girth_from_parity_check(H: np.ndarray):
    """
    Given a parity-check matrix H (M x N),
    build its bipartite adjacency list, then compute and return the girth.
    """
    # 1) Build adjacency
    adjacency_list = build_bipartite_adjacency(H)
    
    # 2) Compute girth
    g = girth_of_bipartite(adjacency_list)
    return g