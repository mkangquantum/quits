"""Validity checks for CSS syndrome-extraction schedules.

Both functions apply to any CSS code whose qubits follow quits' contiguous
indexing convention (data ``0..n-1``, Z-checks ``n..n+mz-1``, X-checks
``n+mz..n+mz+mx-1``) with the standard edge orientation (Z edges
``(data, zcheck)``, X edges ``(xcheck, data)``) — the layout used by the
zxcoloration- and custom-style strategies. Cardinal-family circuits use a
2-D grid indexing and would need their edges remapped first.
"""

import numpy as np
import stim


def interleaving_parity_violations(layers, hx, hz):
    """Overlapping (x, z) check pairs whose gate ordering is invalid.

    A mixed X/Z ordering is only valid when, for every overlapping pair of an
    X-check and a Z-check, the number of shared data qubits where the X gate
    precedes the Z gate is even — otherwise back-propagating the Z-ancilla's
    measured operator leaves a residual ``Z`` on the ``|+>`` X-ancilla and
    the outcome randomizes. This is the invariant the ``cardinal``-style
    direction brackets satisfy by construction.

    Args:
        layers: entangling layers in time order; each a list of oriented
            ``(control, target)`` pairs. Sequential schedules (e.g. Z half
            then X half) are expressed by concatenating their layers in time
            order.
        hx, hz: the parity-check matrices the schedule must implement.

    Returns:
        List of ``(x_row, z_row)`` pairs where the number of shared data
        qubits with the x-gate before the z-gate is odd. Empty means valid.

    Raises:
        ValueError: if the scheduled edges do not reproduce ``hx``/``hz``.
    """
    n_data = hx.shape[1]
    mx, mz = hx.shape[0], hz.shape[0]
    TX = -np.ones((mx, n_data), dtype=np.int64)
    TZ = -np.ones((mz, n_data), dtype=np.int64)
    for t, layer in enumerate(layers):
        for c, tgt in layer:
            if c >= n_data + mz:                    # (X-anc, data)
                TX[c - n_data - mz, tgt] = t
            elif n_data <= tgt < n_data + mz:       # (data, Z-anc)
                TZ[tgt - n_data, c] = t
            else:
                raise ValueError(f"unrecognized edge ({c}, {tgt})")
    if not np.array_equal((TX >= 0).astype(np.uint8), (np.asarray(hx) != 0).astype(np.uint8)):
        raise ValueError("scheduled X edges do not reproduce hx")
    if not np.array_equal((TZ >= 0).astype(np.uint8), (np.asarray(hz) != 0).astype(np.uint8)):
        raise ValueError("scheduled Z edges do not reproduce hz")

    violations = []
    for x in range(mx):
        shared = (TX[x] >= 0) & (TZ >= 0)
        count = ((TX[x] < TZ) & shared).sum(axis=1)
        for z in np.where(count % 2 == 1)[0]:
            violations.append((x, int(z)))
    return violations


def verify_stabilizer_flows(layers, hx, hz):
    """Check with stim that one interleaved round measures the right stabilizers.

    Builds a single bare round (resets, H bracket, the mixed CX layers,
    ancilla measurements) and asserts, via ``stim.Circuit.has_all_flows``,
    two unsigned flows per stabilizer ``s``: preservation (``s -> s``) and
    extraction (``s -> rec[r]`` — the ancilla's record equals the initial
    eigenvalue of ``s``).
    """
    n_data = hx.shape[1]
    mz, mx = hz.shape[0], hx.shape[0]
    zanc = list(range(n_data, n_data + mz))
    xanc = list(range(n_data + mz, n_data + mz + mx))

    c = stim.Circuit()
    c.append("R", zanc + xanc)
    c.append("H", xanc)
    for layer in layers:
        c.append("CX", [q for e in layer for q in e])
    c.append("H", xanc)
    c.append("M", zanc)
    c.append("M", xanc)

    identity = stim.PauliString(n_data + mz + mx)
    flows = []
    for r in range(mz):
        pauli = stim.PauliString(n_data + mz + mx)
        for q in np.where(hz[r])[0]:
            pauli[int(q)] = "Z"
        flows.append(stim.Flow(input=pauli, output=pauli))
        flows.append(stim.Flow(input=pauli, output=identity, measurements=[r]))
    for r in range(mx):
        pauli = stim.PauliString(n_data + mz + mx)
        for q in np.where(hx[r])[0]:
            pauli[int(q)] = "X"
        flows.append(stim.Flow(input=pauli, output=pauli))
        flows.append(stim.Flow(input=pauli, output=identity, measurements=[mz + r]))
    return c.has_all_flows(flows, unsigned=True)
