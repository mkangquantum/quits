"""Unit tests for quits.gf2_util.group (finite groups and F2[G])."""

import numpy as np
import pytest

from quits.gf2_util import (
    MultTableGroup,
    alternating,
    canonicalize,
    cyclic,
    dagger,
    dihedral,
    direct_product,
    element_order,
    generated_subgroup,
    is_self_dagger,
    left_rep,
    matrix_dagger,
    right_rep,
    ring_add,
    ring_mul,
    symmetric,
)


def _mod2(M):
    return np.asarray(M, dtype=np.int64) % 2


def _random_supports(gd, rng, count, max_weight=3):
    out = []
    for _ in range(count):
        w = int(rng.integers(1, max_weight + 1))
        out.append(tuple(sorted(int(g) for g in rng.choice(gd.n, size=w, replace=False))))
    return out


def test_cyclic_structure():
    G = cyclic(5)
    assert G.n == 5
    assert G.is_abelian
    assert G.identity == 0
    assert element_order(1, G) == 5


def test_symmetric_and_alternating_orders():
    S3 = symmetric(3)
    assert S3.n == 6
    assert not S3.is_abelian
    assert symmetric(4).n == 24
    assert alternating(4).n == 12


def test_dihedral_order_and_nonabelian():
    D = dihedral(5)
    assert D.n == 10
    assert not D.is_abelian


def test_direct_product_c2_c3():
    G = direct_product(cyclic(2), cyclic(3))
    assert G.n == 6
    assert G.is_abelian
    orders = {element_order(g, G) for g in range(G.n)}
    assert orders == {1, 2, 3, 6}  # C2 x C3 is isomorphic to C6
    assert G.structure == "C2 x C3"


def test_group_axioms_full():
    for G in (symmetric(3), dihedral(4)):
        e = G.identity
        for i in range(G.n):
            assert G.mult[e][i] == i
            assert G.mult[i][e] == i
            assert G.mult[i][G.inv[i]] == e
            assert G.mult[G.inv[i]][i] == e
        for i in range(G.n):
            for j in range(G.n):
                for k in range(G.n):
                    assert G.mult[G.mult[i][j]][k] == G.mult[i][G.mult[j][k]]


def test_from_generators_closure_matches_direct_construction():
    direct = symmetric(3)
    generated = MultTableGroup.from_generators([(1, 0, 2), (1, 2, 0)], "S3")
    assert generated.perms == direct.perms
    assert generated.mult == direct.mult
    assert generated.inv == direct.inv


def test_multtablegroup_validation_errors():
    with pytest.raises(ValueError, match="empty element list"):
        MultTableGroup([], "empty")
    with pytest.raises(ValueError, match="does not contain the identity"):
        MultTableGroup([(1, 0, 2), (1, 2, 0)], "no-id")
    with pytest.raises(ValueError, match="not closed"):
        MultTableGroup([(0, 1, 2), (1, 0, 2), (0, 2, 1)], "not-closed")
    with pytest.raises(ValueError, match="inconsistent degree"):
        MultTableGroup([(0, 1), (1, 0), (0, 1, 2)], "mixed")


def test_canonicalize_cancels_duplicates_idempotent():
    assert canonicalize([3, 1, 1, 2]) == (2, 3)
    assert canonicalize([]) == ()
    assert canonicalize(canonicalize([5, 5, 5])) == (5,)


def test_ring_axioms():
    G = symmetric(3)
    rng = np.random.default_rng(0)
    xs = _random_supports(G, rng, 6)
    for x, y, z in zip(xs[::3], xs[1::3], xs[2::3]):
        assert ring_add(x, y) == ring_add(y, x)
        assert ring_add(ring_add(x, y), z) == ring_add(x, ring_add(y, z))
        assert ring_add(x, x) == ()
        assert ring_mul(ring_mul(x, y, G), z, G) == ring_mul(x, ring_mul(y, z, G), G)
        lhs = ring_mul(x, ring_add(y, z), G)
        rhs = ring_add(ring_mul(x, y, G), ring_mul(x, z, G))
        assert lhs == rhs


def test_ring_mul_commutativity_matches_group():
    ab = cyclic(6)
    x, y = (0, 2, 3), (1, 5)
    assert ring_mul(x, y, ab) == ring_mul(y, x, ab)
    S3 = symmetric(3)
    # transpositions that do not commute in S3
    assert ring_mul((1,), (2,), S3) != ring_mul((2,), (1,), S3)


def test_dagger_involution_and_antihomomorphism():
    G = symmetric(3)
    rng = np.random.default_rng(1)
    for x, y in zip(_random_supports(G, rng, 4), _random_supports(G, rng, 4)):
        assert dagger(dagger(x, G), G) == canonicalize(x)
        assert dagger(ring_mul(x, y, G), G) == ring_mul(dagger(y, G), dagger(x, G), G)


def test_matrix_dagger_transpose_shape_and_involution():
    G = symmetric(3)
    M = [[(0, 1), (2,)], [(3,), (0, 4, 5)]]
    Md = matrix_dagger(M, G)
    assert len(Md) == 2 and len(Md[0]) == 2
    assert Md[1][0] == dagger(M[0][1], G)
    assert matrix_dagger(matrix_dagger(M, G), G) == [
        [canonicalize(entry) for entry in row] for row in M
    ]


def test_is_self_dagger_examples():
    G = symmetric(3)
    assert is_self_dagger((0,), G)  # identity
    # a 3-cycle is not its own inverse
    three_cycle = next(g for g in range(G.n) if element_order(g, G) == 3)
    assert not is_self_dagger((three_cycle,), G)
    # g + g^-1 is always self-dagger
    assert is_self_dagger((three_cycle, G.inv[three_cycle]), G)


def test_element_order_divides_group_order():
    G = symmetric(4)
    for g in range(G.n):
        assert G.n % element_order(g, G) == 0


def test_generated_subgroup_lagrange():
    G = symmetric(3)
    for g in range(G.n):
        sub = generated_subgroup([g], G)
        assert len(sub) == element_order(g, G)
        assert G.n % len(sub) == 0
    assert generated_subgroup(range(G.n), G) == set(range(G.n))


def test_left_right_rep_homomorphism():
    G = symmetric(3)
    rng = np.random.default_rng(2)
    for x, y in zip(_random_supports(G, rng, 4), _random_supports(G, rng, 4)):
        xy = ring_mul(x, y, G)
        assert np.array_equal(_mod2(left_rep(x, G) @ left_rep(y, G)), left_rep(xy, G))
        assert np.array_equal(_mod2(right_rep(x, G) @ right_rep(y, G)), right_rep(xy, G))


def test_left_right_reps_commute():
    G = symmetric(3)  # non-abelian, so commutation of L and R is non-trivial
    rng = np.random.default_rng(3)
    for x, y in zip(_random_supports(G, rng, 4), _random_supports(G, rng, 4)):
        L, R = left_rep(x, G), right_rep(y, G)
        assert np.array_equal(_mod2(L @ R), _mod2(R @ L))


def test_rep_transpose_is_dagger():
    G = dihedral(4)
    rng = np.random.default_rng(4)
    for x in _random_supports(G, rng, 5):
        xd = dagger(x, G)
        assert np.array_equal(left_rep(x, G).T, left_rep(xd, G))
        assert np.array_equal(right_rep(x, G).T, right_rep(xd, G))


def test_rep_of_identity_and_empty_support():
    G = cyclic(4)
    assert np.array_equal(left_rep((0,), G), np.eye(4, dtype=np.uint8))
    assert np.array_equal(right_rep((0,), G), np.eye(4, dtype=np.uint8))
    assert np.array_equal(left_rep((), G), np.zeros((4, 4), dtype=np.uint8))
