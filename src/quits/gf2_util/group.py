"""Finite groups, the group algebra F2[G], and its regular representations.

The conventions follow the reference implementation of arXiv:2607.28795, so
results cross-check against it. The multiplication table is built from
permutations in pure Python — no GAP install needed.

Conventions:

- Group elements are 0-based indices into a fixed element list, with the
  identity always at index 0.
- A ring element of ``F2[G]`` is stored as a **sorted tuple of distinct
  0-based indices** — just the support, since coefficients are 0/1. All
  public functions accept any iterable of ints and return canonical form;
  repeated entries cancel mod 2.
- Permutations are tuples ``p`` with ``p[i]`` the image of point ``i``, and
  compose as functions: ``(p·q)(i) = p[q[i]]``.
"""

import itertools
from typing import Iterable, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. Permutation helpers
# ─────────────────────────────────────────────────────────────────


def _compose(p: tuple, q: tuple) -> tuple:
    """Composition of permutations: ``(p·q)(i) = p[q[i]]``."""
    return tuple(p[q[i]] for i in range(len(p)))


def _invert(p: tuple) -> tuple:
    """Inverse permutation."""
    out = [0] * len(p)
    for i, x in enumerate(p):
        out[x] = i
    return tuple(out)


def _cycle_str(p: tuple) -> str:
    """Cycle notation for a permutation, 1-based like GAP prints it."""
    seen = [False] * len(p)
    cycles = []
    for start in range(len(p)):
        if seen[start] or p[start] == start:
            seen[start] = True
            continue
        cycle = []
        cur = start
        while not seen[cur]:
            seen[cur] = True
            cycle.append(cur + 1)
            cur = p[cur]
        cycles.append("(" + ",".join(str(c) for c in cycle) + ")")
    return "".join(cycles) if cycles else "()"


# ─────────────────────────────────────────────────────────────────
# 2. Group setup
# ─────────────────────────────────────────────────────────────────


class MultTableGroup:
    """A finite group given by its multiplication table, built from permutations.

    The attributes read by every function below (and by
    :mod:`quits.gf2_util.lifted_product`) are ``n``, ``mult``, ``inv`` and
    ``identity``.

    Attributes:
        n: group order ``|G|``.
        mult: ``mult[i][j]`` = index of ``g_i · g_j``.
        inv: ``inv[i]`` = index of ``g_i⁻¹``.
        identity: index of the identity element — always ``0``.
        perms: the underlying permutation tuples, sorted.
        elem_strs: cycle-notation strings, one per element.
        structure: human-readable name, e.g. ``"C5 x S3"``.
        is_abelian: whether the group is abelian.
    """

    def __init__(self, perms: Iterable[tuple], structure: str):
        perms = sorted(set(tuple(p) for p in perms))
        if not perms:
            raise ValueError("empty element list")
        degree = len(perms[0])
        if any(len(p) != degree for p in perms):
            raise ValueError("permutations have inconsistent degree")
        # The identity is the lexicographically smallest permutation, so
        # sorting alone puts it at index 0 — the convention the companion
        # code enforces.
        if perms[0] != tuple(range(degree)):
            raise ValueError("element list does not contain the identity")

        index = {p: i for i, p in enumerate(perms)}
        self.perms = perms
        self.n = len(perms)
        self.structure = structure
        self.identity = 0
        try:
            self.mult = [[index[_compose(a, b)] for b in perms] for a in perms]
        except KeyError:
            raise ValueError("element list is not closed under composition")
        self.inv = [index[_invert(a)] for a in perms]
        self.elem_strs = [_cycle_str(p) for p in perms]
        self.is_abelian = all(
            self.mult[i][j] == self.mult[j][i]
            for i in range(self.n)
            for j in range(i + 1, self.n)
        )

    @classmethod
    def from_generators(cls, generators: Sequence[tuple], structure: str
                        ) -> "MultTableGroup":
        """Close a set of permutation generators under composition (BFS)."""
        generators = [tuple(g) for g in generators]
        degree = len(generators[0])
        elements = {tuple(range(degree))}
        frontier = list(elements)
        while frontier:
            nxt = []
            for p in frontier:
                for g in generators:
                    q = _compose(g, p)
                    if q not in elements:
                        elements.add(q)
                        nxt.append(q)
            frontier = nxt
        return cls(elements, structure)

    def __repr__(self):
        return (f"MultTableGroup(structure={self.structure!r}, n={self.n}, "
                f"is_abelian={self.is_abelian})")

    def print_elements(self):
        print(f"Elements of {self.structure} (n={self.n}):")
        for i, s in enumerate(self.elem_strs):
            print(f"  g[{i:2d}] = {s:<16s}  inv = {self.elem_strs[self.inv[i]]}")


# ─────────────────────────────────────────────────────────────────
# 3. Named group constructors
# ─────────────────────────────────────────────────────────────────


def cyclic(m: int) -> MultTableGroup:
    """Cyclic group ``C_m`` of order ``m``, acting on ``m`` points."""
    return MultTableGroup.from_generators(
        [tuple((i + 1) % m for i in range(m))], f"C{m}"
    )


def symmetric(m: int) -> MultTableGroup:
    """Symmetric group ``S_m`` of order ``m!``, acting on ``m`` points."""
    return MultTableGroup(itertools.permutations(range(m)), f"S{m}")


def alternating(m: int) -> MultTableGroup:
    """Alternating group ``A_m`` of order ``m!/2``, acting on ``m`` points."""
    def is_even(p):
        # Parity = (number of points) - (number of cycles), mod 2.
        seen = [False] * len(p)
        cycles = 0
        for start in range(len(p)):
            if seen[start]:
                continue
            cycles += 1
            cur = start
            while not seen[cur]:
                seen[cur] = True
                cur = p[cur]
        return (len(p) - cycles) % 2 == 0

    return MultTableGroup(
        (p for p in itertools.permutations(range(m)) if is_even(p)), f"A{m}"
    )


def dihedral(m: int) -> MultTableGroup:
    """Dihedral group of order ``2m`` — the symmetries of an ``m``-gon.

    Note the naming convention: this is GAP's ``DihedralGroup(2*m)``. The
    paper's Table 1 writes ``D_10`` for the group of order 10, which is
    ``dihedral(5)`` here.
    """
    rotation = tuple((i + 1) % m for i in range(m))
    reflection = tuple((-i) % m for i in range(m))
    return MultTableGroup.from_generators([rotation, reflection], f"D{2 * m}")


def direct_product(g1: MultTableGroup, g2: MultTableGroup,
                   structure: str | None = None) -> MultTableGroup:
    """Direct product ``G1 × G2``, acting on the disjoint union of the points."""
    d1 = len(g1.perms[0])
    perms = [p + tuple(x + d1 for x in q) for p in g1.perms for q in g2.perms]
    return MultTableGroup(
        perms, structure or f"{g1.structure} x {g2.structure}"
    )


# ─────────────────────────────────────────────────────────────────
# 4. Ring element operations — F2[G] arithmetic
# ─────────────────────────────────────────────────────────────────


def canonicalize(x: Iterable[int]) -> tuple:
    """Reduce an F2[G] element to canonical form.

    Returns a sorted tuple of distinct indices with even-count entries dropped
    (mod-2 cancellation). Idempotent on already-canonical input.
    """
    seen: set = set()
    for g in x:
        if g in seen:
            seen.discard(g)
        else:
            seen.add(g)
    return tuple(sorted(seen))


def ring_add(x: Iterable[int], y: Iterable[int]) -> tuple:
    """Add two ring elements in F2[G]: mod-2 sum of supports."""
    result: set = set()
    for g in itertools.chain(x, y):
        if g in result:
            result.discard(g)
        else:
            result.add(g)
    return tuple(sorted(result))


def ring_mul(x: Iterable[int], y: Iterable[int], gd: MultTableGroup) -> tuple:
    """Multiply two ring elements in F2[G].

    ``(Σ α_g g) · (Σ β_h h) = Σ_{g,h} α_g β_h (g·h)`` (mod 2). Commutative
    iff ``G`` is abelian.
    """
    result: set = set()
    for g in x:
        for h in y:
            k = gd.mult[g][h]
            if k in result:
                result.discard(k)
            else:
                result.add(k)
    return tuple(sorted(result))


def dagger(x: Iterable[int], gd: MultTableGroup) -> tuple:
    """Involution ``x* = Σ_g α_g g⁻¹`` — invert each element of the support."""
    return canonicalize(gd.inv[g] for g in x)


def matrix_dagger(M: list, gd: MultTableGroup) -> list:
    """Conjugate transpose of a ring matrix: ``M*[j][i] = M[i][j]*``."""
    ma = len(M)
    na = len(M[0])
    return [[dagger(M[ia][ja], gd) for ia in range(ma)] for ja in range(na)]


def is_self_dagger(x: Iterable[int], gd: MultTableGroup) -> bool:
    """Whether ``x = x*``, i.e. ``L[x]`` and ``R[x]`` are symmetric matrices."""
    return dagger(x, gd) == canonicalize(x)


def element_order(g: int, gd: MultTableGroup) -> int:
    """Order of group element ``g``: smallest ``k ≥ 1`` with ``g^k = e``."""
    if g == gd.identity:
        return 1
    k = 1
    current = g
    while current != gd.identity:
        current = gd.mult[current][g]
        k += 1
    return k


def generated_subgroup(elements: Iterable[int], gd: MultTableGroup) -> set:
    """Indices of the subgroup generated by ``elements``."""
    subgroup = {gd.identity}
    frontier = [gd.identity]
    gens = list(elements)
    while frontier:
        nxt = []
        for h in frontier:
            for g in gens:
                k = gd.mult[g][h]
                if k not in subgroup:
                    subgroup.add(k)
                    nxt.append(k)
        frontier = nxt
    return subgroup


# ─────────────────────────────────────────────────────────────────
# 5. Regular representations — |G| × |G| binary matrices
# ─────────────────────────────────────────────────────────────────


def left_rep(x: Iterable[int], gd: MultTableGroup) -> np.ndarray:
    """``L[x]``: left regular representation of ``x ∈ F2[G]``.

    ``L[g]·b(h) = b(g·h)``, i.e. column ``h`` gets a 1 in row ``g·h``. In
    coordinates ``L[a]_{y,x} = α_{y x⁻¹}``.
    """
    n = gd.n
    M = np.zeros((n, n), dtype=np.uint8)
    for g in x:
        for h in range(n):
            M[gd.mult[g][h], h] ^= 1
    return M


def right_rep(x: Iterable[int], gd: MultTableGroup) -> np.ndarray:
    """``R[x]``: right regular representation of ``x ∈ F2[G]``.

    ``R[g]·b(h) = b(h·g⁻¹)``, i.e. column ``h`` gets a 1 in row ``h·g⁻¹``. In
    coordinates ``R[a]_{y,x} = α_{y⁻¹ x}``.

    The ``g⁻¹`` in this convention is what makes the two representations
    commute and makes transposition act as the involution:

    - ``R[x]·R[y] = R[x·y]`` (a homomorphism, like ``L``),
    - ``L[x]·R[y] = R[y]·L[x]``,
    - ``L[x]ᵀ = L[x*]`` and ``R[x]ᵀ = R[x*]``.

    Together these give ``Hx · Hzᵀ = 0`` for the lifted product.
    """
    n = gd.n
    M = np.zeros((n, n), dtype=np.uint8)
    for g in x:
        ginv = gd.inv[g]
        for h in range(n):
            M[gd.mult[h][ginv], h] ^= 1
    return M
