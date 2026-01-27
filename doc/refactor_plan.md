# Refactor Plan (Updated with Completed Changes)

This plan captures the key refactors and API changes implemented so far.

## Summary of Key Revisions

- Added modular circuit-construction strategies under `qldpc_code/circuit_construction/`.
- Introduced `build_circuit(...)` as the primary entrypoint, with `build_graph(...)` retained only as a deprecated wrapper.
- Added new code family: Lift-connected surface code (`LscCode`) and tests.
- Renamed `QlpCode2` to `QlpPolyCode` and updated exports/usages.
- Updated tests and notebooks to use `build_circuit` instead of `build_graph`.
- Added canonical logical controls and verbose logging for HGP/BPC.
- Adjusted CSS logical verification criteria and pairing checks.

## Current Target Layout

```
src/quits/
  __init__.py
  api.py
  circuit.py
  decoder/
    __init__.py
    base.py
    bposd.py
    bplsd.py
    sliding_window.py
  qldpc_code/
    __init__.py
    base.py
    hgp.py
    qlp.py
    bpc.py
    lsc.py
    circuit_construction/
      __init__.py
      base.py
      cardinal.py
      xzcoloration.py
      freeform.py
  gf2_util.py
  ldpc_util.py
  simulation.py
```

## Implemented Changes (Details)

### Circuit construction modularity
- Added `qldpc_code/circuit_construction/` with a strategy registry and base interface.
- Implemented `CardinalBuilder`; added placeholders for `XZColorationBuilder` and `FreeformBuilder`.
- Moved cardinal graph-building helpers into `CardinalBuilder`.
- `QldpcCode.build_circuit()` delegates to the selected builder.
- `build_graph()` now warns via `DeprecationWarning` and calls `build_circuit(strategy="cardinal", ...)`.

### Code-family updates
- `HgpCode`, `QlpCode`, `QlpPolyCode`, and `BpcCode` override `build_circuit(...)` for the cardinal strategy.
- `QlpCode2` renamed to `QlpPolyCode` (exports updated in `qldpc_code/__init__.py` and `api.py`).
- Added `LscCode` in `qldpc_code/lsc.py`, parameterized by `lift_size` (L) and `length` (l+1).

### Canonical logicals + verbose mode
- `HgpCode` and `BpcCode` accept `verbose`; print when canonical logicals are used.
- `HgpCode.get_logicals` renamed to `get_canonical_logicals`.
- `BpcCode.get_logicals` renamed to `get_canonical_logicals`.
- `BpcCode` canonical logicals now:
  - Mix Z logicals (default) to enforce weight 2*q.
  - Re-pair LX/LZ to make pairing identity.
  - Optionally swap to make X logicals weight 2*q (`canonical_weight="x"`).

### CSS logical verification
- `verify_css_logicals` now computes pairing using overlap parity (mod 2).
- `pairing_is_identity` is required for `report["ok"]`.
- Style updated to store `pairing_is_identity` in a local variable like other checks.

### Tests and docs
- Tests now call `build_circuit(...)` (including `tests/test_codes.py`, `tests/test_circuit.py`, etc.).
- Added `LscCode` test inside `tests/test_codes.py`.
- Notebooks updated:
  - All `build_graph` → `build_circuit` in docs.
  - `01_codes_basics.ipynb` includes canonical-logicals explanation + print cells for HGP/BPC.
  - `old_intro.ipynb` updated to `QlpPolyCode`.
- `doc/circuit_distance_search.py` updated to `build_circuit`.

## Compatibility Strategy (Current)

- `build_graph` still exists as a deprecated wrapper to avoid immediate breakage.
- Public imports are maintained via `qldpc_code/__init__.py` and `api.py`.

## Suggested Next Steps

- Implement `XZColorationBuilder` and `FreeformBuilder`.
- Optionally add a small helper to compute full pairing matrices for diagnostics.
- Consider deprecating/removing `build_graph` after a transition period.
