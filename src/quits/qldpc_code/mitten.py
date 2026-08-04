"""Mitten codes: lifted-product CSS codes over F2[G] (arXiv:2607.28795).

A mitten code is ``LP(A, B)`` for ``1 × 2`` base matrices ``A = [a0 a1]``,
``B = [b0 b1]`` over the group algebra ``F2[G]``, with every entry of weight 3,
the identity present in the pivot entries ``a1``/``b1``, and ``L[a1]``,
``R[b1]`` full rank (Definition 2 of the paper). The code has ``n = 5|G|``
data qubits (blocks D1..D5), ``2|G|`` checks of each type, and ``k = |G|``
logical qubits with a canonical conjugate-pair logical basis (Theorem 1).

Three syndrome-extraction strategies are supported by :class:`MittenCode`:

- ``"custom"`` (default) — a depth-12 interleaved mixed X/Z schedule, which
  meets the hard lower bound set by the thumb block's total degree
  ``6 + 6 = 12``. Emitted directly by :meth:`MittenCode.get_custom_circuit`
  as a single interleaved round; drawing goes through the generic
  ``CustomBuilder``.
- ``"hook_error_free"`` — the paper's group-element-layer schedule, depth
  ``12 + 12 = 24``: every layer is a permutation between one check block and
  one data block (one AOD move on an atom array). Emitted through the
  zxcoloration builder with prescribed layers. Appendix H.5 of the paper
  keeps the finger and thumb sections separate and searches over random
  orders of the six layers within each section, screening candidates by
  their estimated circuit-level distance; pass ``seed`` to
  :meth:`MittenCode.build_circuit` to draw such a random order, and screen
  seeds with :func:`quits.simulation.spacetime_matrices` and
  :func:`quits.simulation.estimate_spacetime_distance`. Layer order never
  affects depth or validity, only hook-fault structure.
- ``"zxcoloration"`` — quits' generic Tanner-graph edge coloring, depth 18
  (chromatic index 9 per half).

Interleaved-schedule background: a mixed ordering is valid only when, for
every overlapping pair of an X-check and a Z-check, the number of shared data
qubits where the X gate precedes the Z gate is even (see
:func:`~quits.qldpc_code.circuit_construction.schedule_validation.interleaving_parity_violations`).
For mitten codes the shared qubits of
a check pair sit half in one finger block and half in the thumb, paired
one-to-one by the solutions ``(g, h)`` of the group equation ``i = g·j·h``,
so a schedule is valid iff for every ``(g, h)``:
``[XF_g before ZF_h] ⇔ [XT_h before ZT_g]`` (writing ``XF/ZT`` for the
X-finger/Z-thumb units of ``g`` and ``ZF/XT`` for the Z-finger/X-thumb units
of ``h``). The depth-12 schedule places the 12 mutually conflicting thumb
units in the pattern ``Z X Z X Z X X Z X Z X Z`` and pairs each finger unit
with the *rank-reversed* thumb slot of its side — the reversal complements
every comparison, which is exactly the mirror condition.
"""

import numpy as np
import stim

from ..circuit import Circuit
from ..gf2_util import (
    build_Hx_from_lifted_product,
    build_Hz_from_lifted_product,
    gf2_rank,
    gf2_solve_square,
    left_rep,
    right_rep,
)
from ..noise import ErrorModel
from .base import QldpcCode
from .circuit_construction import get_builder
from .circuit_construction.circuit_build_options import CircuitBuildOptions
from .circuit_construction.schedule_validation import interleaving_parity_violations
from .circuit_construction.zxcoloration import ZXColorationBuilder

# Thumb interleaving pattern of the depth-12 schedule. Its ZT positions have
# XT-below counts {0,1,2,4,5,6} — a self-complementary multiset, which is what
# makes the rank-reversal pairing satisfy the mirror condition.
_DEPTH12_PATTERN = ("Z", "X", "Z", "X", "Z", "X", "X", "Z", "X", "Z", "X", "Z")


# ─────────────────────────────────────────────────────────────────
# 1. The mitten code family
# ─────────────────────────────────────────────────────────────────


def is_valid_mitten(A, B, gd):
    """Whether ``(A, B)`` satisfies Definition 2 of arXiv:2607.28795.

    Requires ``A = [a0 a1]`` and ``B = [b0 b1]`` of shape ``1 × 2`` over
    ``F2[G]``, every entry of weight 3, the identity present in ``a1`` and
    ``b1`` (the canonical form ``a1 = e + g4 + g5``), and ``L[a1]``, ``R[b1]``
    of full rank over GF(2) — the special case of the square invertibility
    condition that gives the canonical logical basis.
    """
    if len(A) != 1 or len(A[0]) != 2 or len(B) != 1 or len(B[0]) != 2:
        raise ValueError("mitten base matrices must have shape 1 x 2")
    (a0, a1), (b0, b1) = A[0], B[0]
    if any(len(x) != 3 for x in (a0, a1, b0, b1)):
        return False
    if gd.identity not in a1 or gd.identity not in b1:
        return False
    return (gf2_rank(left_rep(a1, gd)) == gd.n
            and gf2_rank(right_rep(b1, gd)) == gd.n)


def sample_mitten_base_matrices(gd, rng, max_trials=10_000):
    """Draw random weight-3 base matrices until they define a mitten code.

    Samples ``a0`` and ``b0`` as three distinct group elements, and ``a1``,
    ``b1`` as the identity plus two distinct non-identity elements, then keeps
    the first pair passing :func:`is_valid_mitten`.

    Args:
        gd: the group ``G``.
        rng: a ``numpy.random.Generator``.
        max_trials: how many draws to attempt before giving up.

    Returns:
        ``(A, B)`` as ``1 × 2`` ring matrices.

    Raises:
        RuntimeError: if no valid pair is found within ``max_trials``.
    """
    non_identity = np.arange(1, gd.n)

    def draw_free():
        return tuple(sorted(int(g) for g in
                            rng.choice(gd.n, size=3, replace=False)))

    def draw_pivot():
        return tuple(sorted([gd.identity] + [
            int(g) for g in rng.choice(non_identity, size=2, replace=False)
        ]))

    for _ in range(max_trials):
        A = [[draw_free(), draw_pivot()]]
        B = [[draw_free(), draw_pivot()]]
        if is_valid_mitten(A, B, gd):
            return A, B
    raise RuntimeError(
        f"no valid mitten base matrices found in {max_trials} trials"
    )


def conjugate_pair_overlaps(Lx, Lz):
    """Support-overlap sizes ``|supp(Lx_i) ∩ supp(Lz_i)|`` for each ``i``."""
    return np.sum((np.asarray(Lx) != 0) & (np.asarray(Lz) != 0), axis=1)


def _matching_edges(rep_block, data_block, check_qubit_base, check_block, n,
                    orientation):
    """Oriented CX pairs for one permutation block of the check matrix.

    ``rep_block[i, j] = 1`` couples check qubit ``check_qubit_base +
    check_block*n + i`` to data qubit ``data_block*n + j``. Orientation follows
    quits' zxcoloration convention: Z edges are ``(data, zcheck)`` (control =
    data), X edges are ``(xcheck, data)`` (control = ancilla).
    """
    edges = []
    for i, j in np.argwhere(rep_block != 0):
        check_q = int(check_qubit_base + check_block * n + i)
        data_q = int(data_block * n + j)
        edges.append((data_q, check_q) if orientation == "Z" else (check_q, data_q))
    return edges


# ─────────────────────────────────────────────────────────────────
# 2. The code class
# ─────────────────────────────────────────────────────────────────


class MittenCode(QldpcCode):
    """Mitten code ``LP(A, B)`` over a finite group ``G`` (arXiv:2607.28795).

    Args:
        A: ``1 × 2`` ring matrix ``[a0 a1]`` over ``F2[G]`` (weight-3 entries).
        B: ``1 × 2`` ring matrix ``[b0 b1]``.
        group: the group ``G`` as a ``quits.gf2_util.MultTableGroup``.

    The pair is checked with :func:`is_valid_mitten` and the logicals are the
    paper's Theorem-1 conjugate-pair basis with ``lx @ lz.T = I``. To build a
    generic ``1 × 2`` lifted product instead, pass
    ``build_Hz_from_lifted_product`` / ``build_Hx_from_lifted_product`` output
    to :meth:`QldpcCode.from_parity_checks`.

    Random base matrices can be drawn with
    :func:`sample_mitten_base_matrices`.
    """

    supported_strategies = {"custom", "hook_error_free", "zxcoloration"}

    def __init__(self, A, B, group):
        super().__init__()
        self.A, self.B, self.group = A, B, group
        if not is_valid_mitten(A, B, group):
            raise ValueError(
                "(A, B) is not a valid mitten pair over the given group — "
                "see is_valid_mitten"
            )
        self.hz = build_Hz_from_lifted_product(A, B, group)
        self.hx = build_Hx_from_lifted_product(A, B, group)
        self.lz, self.lx = self.get_canonical_logicals()

    def get_canonical_logicals(self):
        """Canonical logical basis of the mitten code (Theorem 1 of arXiv:2607.28795).

        For each ``g ∈ G``::

            X̄_g = (1_g, u_g, 0, 0, 0)ᵀ,      Z̄_g = (1_g, 0, v_g, 0, 0)ᵀ

        where ``u_g`` and ``v_g`` are the unique solutions of::

            R[b0]·1_g + R[b1]·u_g = 0,       L[a0]·1_g + L[a1]·v_g = 0.

        Uniqueness — hence the whole basis — comes from ``L[a1]`` and
        ``R[b1]`` being full rank. Solving for all ``g`` at once means solving
        with the identity as right-hand side, so ``U = R[b1]⁻¹R[b0]`` has
        ``u_g`` as its ``g``-th column, and likewise ``V = L[a1]⁻¹L[a0]``.

        Returns:
            ``(lz, lx)``, each of shape ``(|G|, 5|G|)``, row ``g`` holding
            ``Z̄_g`` resp. ``X̄_g``. They satisfy ``lx · lzᵀ = I``.
        """
        gd = self.group
        (a0, a1), (b0, b1) = self.A[0], self.B[0]
        n = gd.n

        U = gf2_solve_square(right_rep(b1, gd), right_rep(b0, gd))
        V = gf2_solve_square(left_rep(a1, gd), left_rep(a0, gd))

        identity_block = np.eye(n, dtype=np.uint8)
        zero_block = np.zeros((n, n), dtype=np.uint8)

        # Row g of each basis; blocks are D1 … D5 left to right.
        lx = np.concatenate(
            [identity_block, U.T, zero_block, zero_block, zero_block], axis=1
        )
        lz = np.concatenate(
            [identity_block, zero_block, V.T, zero_block, zero_block], axis=1
        )
        return lz.astype(np.uint8), lx.astype(np.uint8)

    def build_circuit(
        self,
        strategy="custom",
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        **opts,
    ):
        """Build the memory-experiment circuit for the given strategy.

        Strategies: ``"custom"`` (default) — the depth-12 interleaved mixed
        X/Z schedule; ``"hook_error_free"`` — the paper's depth-24
        group-element-layer schedule; ``"zxcoloration"`` — quits' generic
        edge coloring, depth 18.

        For ``"hook_error_free"``, ``seed`` (int) draws a random order of the
        six layers within each finger/thumb section (Appendix H.5's search
        space); omit it for the fixed default order. Other extra keyword
        arguments are accepted and ignored by the native strategies. Sets
        ``self.depth`` as a side effect.
        """
        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")

        if strategy == "custom":
            get_builder("custom", self)
            return self.get_custom_circuit(
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
            )
        elif strategy == "hook_error_free":
            return self._build_hook_error_free_circuit(
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
                seed=opts.get("seed"),
            )
        elif strategy == "zxcoloration":
            builder = get_builder("zxcoloration", self)
            return builder.get_coloration_circuit(
                error_model=error_model,
                num_rounds=num_rounds,
                basis=basis,
                circuit_build_options=circuit_build_options,
            )
        else:
            return super().build_circuit(strategy=strategy, **opts)

    def _ensure_custom_qubit_indexing(self):
        n = int(self.group.n)

        self.data_qubits = np.arange(0, 5 * n, dtype=int)
        self.zcheck_qubits = np.arange(5 * n, 7 * n, dtype=int)
        self.xcheck_qubits = np.arange(7 * n, 9 * n, dtype=int)
        self.check_qubits = np.concatenate((self.zcheck_qubits, self.xcheck_qubits))
        self.all_qubits = np.arange(9 * n, dtype=int)
        return n

    def get_custom_schedule_edges(self):
        """The depth-12 interleaved mixed X/Z schedule as CX rounds.

        Thumb units follow the ``Z X Z X Z X X Z X Z X Z`` pattern; the finger
        unit of rank ``r`` shares the slot of the rank-``(5-r)`` thumb unit of
        the same side (the rank-reversal pairing). Each round is a perfect
        matching on all ``9|G|`` qubits; validity is asserted before returning.

        Returns:
            Dict ``{"round1": [...], ..., "round12": [...]}`` of oriented
            ``(control, target)`` pairs.
        """
        self._ensure_custom_qubit_indexing()

        XF, ZF, XT, ZT = self._unit_edges()
        zpos = [t for t, p in enumerate(_DEPTH12_PATTERN) if p == "Z"]
        xpos = [t for t, p in enumerate(_DEPTH12_PATTERN) if p == "X"]

        layers = [[] for _ in range(12)]
        for r in range(6):
            layers[zpos[r]] += ZT[r]
            layers[xpos[r]] += XT[r]
            layers[zpos[5 - r]] += XF[r]     # rank reversal: XF_r rides ZT_{5-r}'s slot
            layers[xpos[5 - r]] += ZF[r]
        self._assert_valid_schedule(layers)

        return {f"round{t + 1}": layer for t, layer in enumerate(layers)}

    def get_custom_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
    ):
        """Memory-experiment circuit for the depth-12 interleaved schedule.

        The stabilizer round is a single interleaved block::

            R (all ancillas) → H (X ancillas) → mixed CX rounds → H (X ancillas)
            → M (Z ancillas) → M (X ancillas)

        The measurement order (Z block then X block per round) matches
        zxcoloration's record layout, so the circuits stay compatible with
        quits' sliding-window decoders. Sets ``self.depth`` as a side effect.
        """
        self._ensure_custom_qubit_indexing()

        if error_model is None:
            error_model = ErrorModel()
        if circuit_build_options is None:
            circuit_build_options = CircuitBuildOptions()
        elif not isinstance(circuit_build_options, CircuitBuildOptions):
            raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")

        basis = basis.upper()
        if basis not in ("Z", "X"):
            raise ValueError("basis must be 'Z' or 'X'")
        get_Z_detectors = basis == "Z" or circuit_build_options.get_all_detectors
        get_X_detectors = basis == "X" or circuit_build_options.get_all_detectors
        schedule_edges = self.get_custom_schedule_edges()

        circ = Circuit(self.all_qubits)
        self.depth = len(schedule_edges)

        def flatten(edges):
            return [q for edge in edges for q in edge]

        def _add_stabilizer_round():
            circ.add_reset(self.check_qubits)
            circ.add_tick()
            circ.add_hadamard_layer(self.xcheck_qubits)
            for round_index in range(len(schedule_edges)):
                circ.add_cnot_layer(flatten(schedule_edges[f"round{round_index + 1}"]))
            circ.add_hadamard_layer(self.xcheck_qubits)
            circ.add_measure_layer(self.zcheck_qubits)
            circ.add_measure_layer(self.xcheck_qubits)

        # Logical state prep
        if circuit_build_options.noisy_zeroth_round:
            circ.set_error_model(error_model)
        else:
            circ.set_error_model(ErrorModel.zero())

        circ.add_reset(self.data_qubits, basis)
        _add_stabilizer_round()

        if basis == "Z":
            for i in range(1, len(self.zcheck_qubits) + 1)[::-1]:
                circ.add_detector([len(self.xcheck_qubits) + i])
        else:
            for i in range(1, len(self.xcheck_qubits) + 1)[::-1]:
                circ.add_detector([i])

        # Logical memory with noise
        circ.set_error_model(error_model)

        if num_rounds > 0:
            circ.start_loop(num_rounds)
            _add_stabilizer_round()
            if get_Z_detectors:
                for i in range(1, len(self.zcheck_qubits) + 1)[::-1]:
                    ind = len(self.xcheck_qubits) + i
                    circ.add_detector([ind, ind + len(self.check_qubits)])
            if get_X_detectors:
                for i in range(1, len(self.xcheck_qubits) + 1)[::-1]:
                    circ.add_detector([i, i + len(self.check_qubits)])
            circ.end_loop()

        # Logical measurement
        if not circuit_build_options.noisy_final_meas:
            circ.set_error_model(ErrorModel.zero())

        circ.add_measure(self.data_qubits, basis)

        if basis == "Z":
            for i in range(1, len(self.zcheck_qubits) + 1)[::-1]:
                inds = np.array([len(self.data_qubits) + len(self.xcheck_qubits) + i])
                inds = np.concatenate((inds, len(self.data_qubits) - np.where(
                    self.hz[len(self.zcheck_qubits) - i, :] == 1)[0]))
                circ.add_detector(inds)
            for i in range(len(self.lz)):
                circ.add_observable(i, len(self.data_qubits) - np.where(self.lz[i, :] == 1)[0])
        else:
            for i in range(1, len(self.xcheck_qubits) + 1)[::-1]:
                inds = np.array([len(self.data_qubits) + i])
                inds = np.concatenate((inds, len(self.data_qubits) - np.where(
                    self.hx[len(self.xcheck_qubits) - i, :] == 1)[0]))
                circ.add_detector(inds)
            for i in range(len(self.lx)):
                circ.add_observable(i, len(self.data_qubits) - np.where(self.lx[i, :] == 1)[0])

        return stim.Circuit(circ.circuit)

    def _build_hook_error_free_circuit(
        self,
        error_model=None,
        num_rounds=0,
        basis="Z",
        circuit_build_options=None,
        seed=None,
    ):
        """Depth-24 group-element-layer schedule, emitted as two halves.

        Per half, the two check blocks act in parallel on disjoint finger
        blocks for 6 layers (3 per ring entry) and then serialize on the thumb
        block D5 for 6 more. Qubit indexing follows quits' zxcoloration
        builder: data qubits ``0 .. 5n-1`` (blocks D1..D5), Z-check qubits
        ``5n .. 7n-1`` (Z0 then Z1, the row order of
        ``build_Hz_from_lifted_product``), X-check qubits ``7n .. 9n-1``.

        ``seed=None`` keeps the fixed default layer order. An int ``seed``
        applies four independent random permutations — one per finger/thumb
        section of each half — which is the search space of the paper's
        hook-error schedule optimization (Appendix H.5); the sections
        themselves stay separate, so depth and validity are unaffected.
        """
        gd = self.group
        (a0, a1), (b0, b1) = self.A[0], self.B[0]
        n = gd.n
        z_base, x_base = 5 * n, 7 * n

        # Z half.  Hz = [ R(b0) R(b1) 0 0 | L(a0*) ]  row Z0
        #               [ 0 0 R(b0) R(b1) | L(a1*) ]  row Z1
        layers_Z = []
        for col, data_pair in [(b0, (0, 2)), (b1, (1, 3))]:   # (D1,D3) then (D2,D4)
            for g in col:                                      # fingers: Z0 ∥ Z1
                R = right_rep((g,), gd)
                layers_Z.append(
                    _matching_edges(R, data_pair[0], z_base, 0, n, "Z")
                    + _matching_edges(R, data_pair[1], z_base, 1, n, "Z")
                )
        for check_block, col in [(0, a0), (1, a1)]:            # thumb: serialized
            for g in col:
                L = left_rep((gd.inv[g],), gd)                 # per-element L(g⁻¹) of L(a*)
                layers_Z.append(_matching_edges(L, 4, z_base, check_block, n, "Z"))

        # X half.  Hx = [ L(a0) 0 L(a1) 0 | R(b0*) ]  row X0
        #               [ 0 L(a0) 0 L(a1) | R(b1*) ]  row X1
        layers_X = []
        for col, data_pair in [(a0, (0, 1)), (a1, (2, 3))]:   # (D1,D2) then (D3,D4)
            for g in col:                                      # fingers: X0 ∥ X1
                L = left_rep((g,), gd)
                layers_X.append(
                    _matching_edges(L, data_pair[0], x_base, 0, n, "X")
                    + _matching_edges(L, data_pair[1], x_base, 1, n, "X")
                )
        for check_block, col in [(0, b0), (1, b1)]:            # thumb: serialized
            for g in col:
                R = right_rep((gd.inv[g],), gd)                # per-element R(g⁻¹) of R(b*)
                layers_X.append(_matching_edges(R, 4, x_base, check_block, n, "X"))

        if seed is not None:
            rng = np.random.default_rng(seed)

            def shuffle_sections(layers):
                fingers, thumbs = layers[:6], layers[6:]
                return ([fingers[i] for i in rng.permutation(6)]
                        + [thumbs[i] for i in rng.permutation(6)])

            layers_Z = shuffle_sections(layers_Z)
            layers_X = shuffle_sections(layers_X)

        builder = ZXColorationBuilder(self, layers_Z=layers_Z, layers_X=layers_X)
        return builder.get_coloration_circuit(
            error_model=error_model,
            num_rounds=num_rounds,
            basis=basis,
            circuit_build_options=circuit_build_options,
        )

    def _unit_edges(self):
        """The 24 group-element units as oriented edge lists, keyed by kind.

        ``XF[k]``/``ZT[k]`` are indexed by the flat A-side unit list
        ``[(ja, g) for ja in (0,1) for g in a_ja]`` and ``ZF[k]``/``XT[k]`` by
        the B-side list, so ``XF[k]`` and ``ZT[k]`` carry the same group
        element. ``XF[k] ∥ ZT[j]`` are qubit-disjoint (fingers touch D1..D4
        and X ancillas; thumbs touch D5 and Z ancillas), likewise
        ``ZF[k] ∥ XT[j]``.
        """
        gd = self.group
        n = gd.n
        zb, xb = 5 * n, 7 * n

        def edges(rep, dblk, cbase, cblk, orient):
            out = []
            for i, d in np.argwhere(rep != 0):
                cq, dq = int(cbase + cblk * n + i), int(dblk * n + d)
                out.append((dq, cq) if orient == "Z" else (cq, dq))
            return out

        UA = [(ja, g) for ja in (0, 1) for g in self.A[0][ja]]
        UB = [(jb, h) for jb in (0, 1) for h in self.B[0][jb]]
        XF = [edges(left_rep((g,), gd), 2 * ja + 0, xb, 0, "X")
              + edges(left_rep((g,), gd), 2 * ja + 1, xb, 1, "X") for ja, g in UA]
        ZT = [edges(left_rep((gd.inv[g],), gd), 4, zb, ja, "Z") for ja, g in UA]
        ZF = [edges(right_rep((h,), gd), 0 + jb, zb, 0, "Z")
              + edges(right_rep((h,), gd), 2 + jb, zb, 1, "Z") for jb, h in UB]
        XT = [edges(right_rep((gd.inv[h],), gd), 4, xb, jb, "X") for jb, h in UB]
        return XF, ZF, XT, ZT

    def _assert_valid_schedule(self, layers):
        for t, layer in enumerate(layers):
            qs = [q for e in layer for q in e]
            if len(qs) != len(set(qs)):
                raise AssertionError(f"layer {t} is not a matching")
        violations = interleaving_parity_violations(layers, self.hx, self.hz)
        if violations:
            raise AssertionError(
                f"schedule violates the interleaving parity condition on "
                f"{len(violations)} check pairs"
            )
