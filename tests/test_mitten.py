import numpy as np
import pytest
import stim
from ldpc.bposd_decoder import BpOsdDecoder

from quits.circuit import check_overlapping_CX
from quits.decoder.sliding_window import sliding_window_phenom_mem
from quits.gf2_util import (
    build_Hx_from_lifted_product,
    build_Hz_from_lifted_product,
    cyclic,
    dihedral,
    direct_product,
    left_rep,
    right_rep,
    symmetric,
)
from quits.noise import ErrorModel
from quits.qldpc_code import MittenCode
from quits.qldpc_code.circuit_construction import (
    get_builder,
    interleaving_parity_violations,
    verify_stabilizer_flows,
)
from quits.qldpc_code.mitten import (
    conjugate_pair_overlaps,
    is_valid_mitten,
    sample_mitten_base_matrices,
)
from quits.qldpc_code.circuit_construction.zxcoloration import ZXColorationBuilder
from quits.simulation import (
    estimate_spacetime_distance,
    get_stim_mem_result,
    spacetime_matrices,
)

# Valid mitten pair over cyclic(4) -> [[20, 4]] code (fixtures shared with
# tests/test_gf2_lifted_product.py).
C4_A = [[(0, 1, 3), (0, 1, 3)]]
C4_B = [[(0, 2, 3), (0, 1, 2)]]

# Valid lifted-product pair over S3 that is NOT a valid mitten pair.
S3_A = [[(0, 1, 3), (0, 2, 4)]]
S3_B = [[(0, 1, 2), (0, 3, 5)]]

STRATEGY_DEPTHS = {"custom": 12, "zxcoloration": 18, "hook_error_free": 24}


def _c4_code(**kwargs):
    return MittenCode(C4_A, C4_B, cyclic(4), **kwargs)


def _extract_cx_layers(circuit):
    """The entangling layers of one stabilizer round, from the REPEAT block.

    Each CX instruction in the round body is one layer; returns a list of
    layers, each a list of (control, target) pairs in time order.
    """
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            layers = []
            for i in inst.body_copy():
                if i.name == "CX":
                    vals = [t.value for t in i.targets_copy()]
                    layers.append(list(zip(vals[0::2], vals[1::2])))
            return layers
    raise AssertionError("no REPEAT block found")


def _depth18_bracket_schedule(code):
    """The cardinal-style bracket ``XF < (ZF ∥ XT) < ZT`` — 18 layers.

    Valid for any mitten code by the bracket argument (both members of every
    solution pair see the same relative X/Z order).
    """
    XF, ZF, XT, ZT = code._unit_edges()
    return ([list(u) for u in XF]
            + [ZF[k] + XT[k] for k in range(6)]
            + [list(u) for u in ZT])


def _overlap_solution_structure(A, B, gd):
    """Verify the mitten overlap lemma and return the overlap histogram.

    For every pair (X-check ``(ib, i)``, Z-check ``(ja, j)``), the number of
    shared data qubits in the finger block equals the number in the thumb,
    and both equal the number of solutions ``(g, h)`` of ``i = g·j·h`` with
    ``g`` in ``a_ja`` and ``h`` in ``b_ib``.
    """
    n = gd.n
    hz = build_Hz_from_lifted_product(A, B, gd)
    hx = build_Hx_from_lifted_product(A, B, gd)
    supx = [set(np.where(hx[r])[0]) for r in range(2 * n)]
    supz = [set(np.where(hz[r])[0]) for r in range(2 * n)]
    lemma_holds, hist = True, {}
    for x in range(2 * n):
        ib, i = divmod(x, n)
        for z in range(2 * n):
            ja, j = divmod(z, n)
            shared = supx[x] & supz[z]
            finger = sum(1 for q in shared if q < 4 * n)
            thumb = len(shared) - finger
            n_sol = sum(1 for g in A[0][ja] for h in B[0][ib]
                        if gd.mult[gd.mult[g][j]][h] == i)
            if not (finger == thumb == n_sol):
                lemma_holds = False
            hist[len(shared)] = hist.get(len(shared), 0) + 1
    return {"lemma_holds": lemma_holds, "overlap_histogram": dict(sorted(hist.items()))}


def test_is_valid_mitten_true_and_false_branches():
    C4 = cyclic(4)
    assert is_valid_mitten(C4_A, C4_B, C4)
    # entry of weight 2
    assert not is_valid_mitten([[(0, 1), (0, 1, 3)]], C4_B, C4)
    # pivot entry without the identity
    assert not is_valid_mitten([[(0, 1, 3), (1, 2, 3)]], C4_B, C4)
    # rank-deficient pivot: over C3 the only weight-3 pivot is the whole
    # group, whose left representation is the all-ones matrix (rank 1)
    C3 = cyclic(3)
    assert not is_valid_mitten([[(0, 1, 2), (0, 1, 2)]], [[(0, 1, 2), (0, 1, 2)]], C3)
    with pytest.raises(ValueError, match="shape 1 x 2"):
        is_valid_mitten([[(0, 1, 3)]], C4_B, C4)


def test_sample_mitten_seeded_deterministic():
    D4 = dihedral(4)
    A, B = sample_mitten_base_matrices(D4, np.random.default_rng(1))
    assert A == [[(2, 3, 6), (0, 1, 6)]]
    assert B == [[(1, 2, 6), (0, 2, 5)]]
    assert is_valid_mitten(A, B, D4)


def test_sample_mitten_exhausts_trials():
    # no valid mitten pair exists over C3 (see rank-deficiency test above)
    with pytest.raises(RuntimeError, match="no valid mitten base matrices"):
        sample_mitten_base_matrices(cyclic(3), np.random.default_rng(0), max_trials=50)


def test_construction_shapes_and_params():
    code = _c4_code()
    assert code.hx.shape == (8, 20)
    assert code.hz.shape == (8, 20)
    assert code.lx.shape == (4, 20)   # k = |G| = 4
    assert code.lz.shape == (4, 20)
    assert code.A == C4_A and code.B == C4_B
    assert code.group.n == 4


def test_canonical_logicals_pass_verifier_and_pairing():
    code = _c4_code()
    assert code.verify_css_logicals()["all_tests_passed"]
    assert np.array_equal((code.lx @ code.lz.T) % 2, np.eye(4, dtype=np.uint8))


def test_canonical_logicals_theorem1():
    C4 = cyclic(4)
    n = C4.n
    code = _c4_code()
    assert not np.any((code.hz.astype(np.int64) @ code.lx.T.astype(np.int64)) % 2)
    assert not np.any((code.hx.astype(np.int64) @ code.lz.T.astype(np.int64)) % 2)

    # the defining equations of Theorem 1: U, V sit transposed in blocks D2, D3
    (a0, a1), (b0, b1) = C4_A[0], C4_B[0]
    U = code.lx[:, n:2 * n].T
    V = code.lz[:, 2 * n:3 * n].T
    assert np.array_equal(
        (right_rep(b1, C4).astype(np.int64) @ U.astype(np.int64)) % 2,
        right_rep(b0, C4),
    )
    assert np.array_equal(
        (left_rep(a1, C4).astype(np.int64) @ V.astype(np.int64)) % 2,
        left_rep(a0, C4),
    )


def test_canonical_logicals_golden():
    code = _c4_code()
    expected_lx = np.array([
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    expected_lz = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    assert np.array_equal(code.lx, expected_lx)
    assert np.array_equal(code.lz, expected_lz)


def test_conjugate_pair_overlaps_known():
    code = _c4_code()
    assert np.array_equal(conjugate_pair_overlaps(code.lx, code.lz),
                          np.array([1, 1, 1, 1]))
    # disjoint supports
    assert np.array_equal(
        conjugate_pair_overlaps(np.array([[1, 0]]), np.array([[0, 1]])),
        np.array([0]),
    )


def test_invalid_pair_raises():
    with pytest.raises(ValueError, match="not a valid mitten pair"):
        MittenCode(S3_A, S3_B, symmetric(3))


@pytest.mark.parametrize("strategy,depth", sorted(STRATEGY_DEPTHS.items()))
def test_depths_per_strategy(strategy, depth):
    code = _c4_code()
    circuit = code.build_circuit(strategy=strategy, num_rounds=2)
    assert code.depth == depth
    assert len(_extract_cx_layers(circuit)) == depth


def test_custom_circuit_layers_valid():
    code = _c4_code()
    circuit = code.build_circuit(strategy="custom", num_rounds=1)
    layers = _extract_cx_layers(circuit)
    assert len(layers) == 12
    for layer in layers:
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))   # every layer is a matching
    assert interleaving_parity_violations(layers, code.hx, code.hz) == []
    # CustomBuilder populates the Tanner-edge bookkeeping lazily (for drawing)
    get_builder("custom", code).build_graph()
    assert {e for layer in layers for e in layer} == set(code.edges_Z) | set(code.edges_X)
    assert code.custom_num_colors == 12
    assert verify_stabilizer_flows(layers, code.hx, code.hz)


def test_custom_schedule_edges_contract():
    # the round-dict contract CustomBuilder relies on for graph drawing
    code = _c4_code()
    schedule_edges = code.get_custom_schedule_edges()
    assert set(schedule_edges) == {f"round{i}" for i in range(1, 13)}
    for layer in schedule_edges.values():
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))


def test_hook_error_free_circuit_layers_valid():
    code = _c4_code()
    circuit = code.build_circuit(strategy="hook_error_free", num_rounds=1)
    layers = _extract_cx_layers(circuit)
    assert len(layers) == 24
    for layer in layers:
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))
    # the round runs the Z half first, then the X half
    layers_Z, layers_X = layers[:12], layers[12:]
    assert {e for layer in layers_Z for e in layer} == set(code.edges_Z)
    assert {e for layer in layers_X for e in layer} == set(code.edges_X)
    # the sequential two-half schedule trivially satisfies the parity condition
    assert interleaving_parity_violations(layers, code.hx, code.hz) == []


def test_hook_error_free_seed_shuffles_within_sections():
    # seed draws random within-section layer orders (the Appendix H.5 search
    # space); the layer *set* and the section split are unchanged, so depth,
    # validity, and detector determinism are unaffected.
    default = _extract_cx_layers(_c4_code().build_circuit(strategy="hook_error_free", num_rounds=2))
    seeded = _extract_cx_layers(
        _c4_code().build_circuit(strategy="hook_error_free", num_rounds=2, seed=0)
    )
    seeded_again = _extract_cx_layers(
        _c4_code().build_circuit(strategy="hook_error_free", num_rounds=2, seed=0)
    )
    assert seeded == seeded_again              # reproducible
    assert seeded != default                   # order actually changed
    as_sets = lambda layers: {frozenset(layer) for layer in layers}
    # same 24 layers overall, and the Z half (first 12) keeps its layer set
    assert as_sets(seeded) == as_sets(default)
    assert as_sets(seeded[:12]) == as_sets(default[:12])

    code = _c4_code()
    circuit = code.build_circuit(strategy="hook_error_free", num_rounds=2, seed=0)
    assert code.depth == 24
    assert not circuit.compile_detector_sampler().sample(shots=8).any()


def test_naive_pairing_has_violations():
    # Pair each finger unit with the same-rank thumb slot (no rank reversal):
    # the mirror condition fails and the parity checker must flag it.
    code = _c4_code()
    XF, ZF, XT, ZT = code._unit_edges()
    naive = [XF[k] + ZT[k] for k in range(6)] + [ZF[k] + XT[k] for k in range(6)]
    assert interleaving_parity_violations(naive, code.hx, code.hz) != []


def test_depth18_bracket_valid():
    code = _c4_code()
    layers = _depth18_bracket_schedule(code)
    assert len(layers) == 18
    assert interleaving_parity_violations(layers, code.hx, code.hz) == []
    assert verify_stabilizer_flows(layers, code.hx, code.hz)


@pytest.mark.parametrize(
    "strategy,basis",
    [("custom", "Z"), ("custom", "X"), ("zxcoloration", "Z"), ("hook_error_free", "Z")],
)
def test_noiseless_detectors_all_zero(strategy, basis):
    code = _c4_code()
    circuit = code.build_circuit(strategy=strategy, num_rounds=2, basis=basis)
    samples = circuit.compile_detector_sampler().sample(shots=8)
    assert not samples.any()


@pytest.mark.parametrize("strategy", sorted(STRATEGY_DEPTHS))
def test_no_overlapping_cx(strategy):
    p = 1e-3
    code = _c4_code()
    circuit = code.build_circuit(
        strategy=strategy, error_model=ErrorModel(p, p, p, p), num_rounds=1
    )
    assert check_overlapping_CX(circuit) == []


def test_strategy_dispatch_errors():
    code = _c4_code()
    with pytest.raises(NotImplementedError):
        code.build_circuit(strategy="cardinal")
    with pytest.raises(NotImplementedError):
        code.build_circuit(strategy="cardinalNSmerge")
    with pytest.raises(ValueError):
        code.build_circuit(strategy="nonsense")
    # extra kwargs like seed are accepted and ignored by native strategies
    code.build_circuit(strategy="custom", seed=7)


def test_num_colors_bookkeeping():
    code = _c4_code()
    code.build_circuit(strategy="hook_error_free")
    assert code.num_colors == {"Z": 12, "X": 12}
    code = _c4_code()
    code.build_circuit(strategy="custom")
    assert code.depth == 12


def test_zxcoloration_prescribed_layers_validation():
    code = _c4_code()
    circuit = code.build_circuit(strategy="hook_error_free", num_rounds=1)
    layers = _extract_cx_layers(circuit)
    layers_Z, layers_X = layers[:12], layers[12:]
    with pytest.raises(ValueError, match="given together"):
        ZXColorationBuilder(_c4_code(), layers_Z=layers_Z)
    with pytest.raises(ValueError, match="does not reproduce the Z Tanner edges"):
        ZXColorationBuilder(_c4_code(), layers_Z=layers_Z[:-1], layers_X=layers_X)


def test_mitten_lfr_end_to_end():
    # The C4 toy code has weight-2 canonical logicals (distance <= 2), so the
    # memory test uses a [[150, 30]] mitten code over C5 x S3 instead.
    p, num_rounds, num_trials, W, F = 1e-3, 5, 50, 5, 3
    group = direct_product(cyclic(5), symmetric(3))
    A, B = sample_mitten_base_matrices(group, np.random.default_rng(2))
    code = MittenCode(A, B, group)
    assert code.verify_css_logicals()["all_tests_passed"]
    circuit = code.build_circuit(
        strategy="custom",
        error_model=ErrorModel(p, p, p, p),
        num_rounds=num_rounds,
        basis="Z",
    )
    detection_events, observable_flips = get_stim_mem_result(circuit, num_trials, seed=1)

    decoder_params = {
        "bp_method": "product_sum",
        "max_iter": 10,
        "schedule": "serial",
        "osd_method": "osd_cs",
        "osd_order": 1,
        "error_rate": float(p * (code.depth + 3)),
    }
    logical_pred = sliding_window_phenom_mem(
        detection_events,
        code.hz,
        code.lz,
        W,
        F,
        BpOsdDecoder,
        BpOsdDecoder,
        decoder_params,
        decoder_params,
        "decode",
        "decode",
        tqdm_on=False,
    )
    pL = np.mean((observable_flips - logical_pred).any(axis=1))
    lfr = 1 - (1 - pL) ** (1 / num_rounds)
    print(f"MITTEN interleaved [[150,30]] pL={pL:.4f} lfr={lfr:.4f}")
    assert pL <= 0.3
    assert lfr <= 0.1


def test_spacetime_matrices_and_distance():
    code = _c4_code()
    circuit = code.build_circuit(
        strategy="custom", error_model=ErrorModel(1e-3, 1e-3, 1e-3, 1e-3), num_rounds=1
    )
    H, L = spacetime_matrices(circuit)
    assert H.shape[0] == circuit.num_detectors
    assert L.shape[0] == circuit.num_observables == 4
    assert H.shape[1] == L.shape[1] > 0
    d = estimate_spacetime_distance(H, L, num_trials=3, seed=0, osd_order=0)
    assert isinstance(d, int)
    assert d >= 2


def test_overlap_solution_structure():
    result = _overlap_solution_structure(C4_A, C4_B, cyclic(4))
    assert result["lemma_holds"] is True
    assert sum(result["overlap_histogram"].values()) == (2 * 4) ** 2
