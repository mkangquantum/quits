# Refactor Plan (No External API Changes)

This plan proposes a structural refactor that keeps the public API stable while improving internal modularity and organization.

## Goals

- Preserve existing import paths and public functions/classes.
- Improve maintainability by splitting large modules into focused submodules.
- Add an internal facade layer to provide stable re-exports.
- Keep all current behavior and signatures intact.

## Proposed Target Layout

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
  gf2_util.py
  ldpc_util.py
  simulation.py
```

Notes:
- `gf2_util.py` and `ldpc_util.py` remain in place to preserve current imports.
- New `decoder/` and `qldpc_code/` packages hold the refactored implementations.
- `api.py` (or `__init__.py`) becomes the stable user-facing entry point.

## Detailed Steps

### 1. Add a facade API layer

- Create `src/quits/api.py` with re-exports of the most commonly used classes and functions.
- Update `src/quits/__init__.py` to import and re-export from `api.py`.
- This allows internal module changes without affecting user imports.

### 2. Refactor `qldpc_code.py` into a package

- Create `src/quits/qldpc_code/` package.
- Move the existing `QldpcCode` base class and shared helpers into `qldpc_code/base.py`.
- Move concrete code families into separate files:
  - `qldpc_code/hgp.py`
  - `qldpc_code/qlp.py`
  - `qldpc_code/bpc.py`
- Add `qldpc_code/__init__.py` to re-export public classes.
- Keep `src/quits/qldpc_code.py` as a thin compatibility shim that re-exports from the new package.

### 3. Refactor `decoder.py` into a package

- Create `src/quits/decoder/` package.
- Extract shared interfaces into `decoder/base.py` (e.g., `DecoderProtocol`, configs).
- Extract decoder integrations into separate files:
  - `decoder/bposd.py`
  - `decoder/bplsd.py`
- Move the sliding-window algorithm into `decoder/sliding_window.py`.
- Add `decoder/__init__.py` to re-export public APIs.
- Keep `src/quits/decoder.py` as a thin compatibility shim that re-exports from the new package.

### 4. Consolidate high-level imports

- Keep all legacy imports working by:
  - Preserving existing module filenames as re-export shims.
  - Using `__all__` and explicit exports in new packages.
- Ensure all current import paths still work (e.g., `from quits.decoder import sliding_window_phenom_mem`).

## Compatibility Strategy

- Use thin shim modules (`qldpc_code.py`, `decoder.py`) that import from the new package and re-export names.
- Avoid renaming public functions or changing signatures.
- Include smoke tests to verify old import paths.

## Suggested Verification (Non-Exhaustive)

- `python -c "from quits.qldpc_code import QldpcCode"`
- `python -c "from quits.decoder import sliding_window_phenom_mem"`
- `python -c "from quits import QldpcCode"` (after adding `api.py`/`__init__.py` re-exports)

## Rollout Notes

- Refactor in small steps to reduce risk:
  1. Add new package skeletons + re-exports.
  2. Move code incrementally per module.
  3. Remove duplication once re-exports are stable.
- Update docs to point to the new preferred imports, while maintaining backwards compatibility.
