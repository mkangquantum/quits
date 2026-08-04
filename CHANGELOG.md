# Changelog

All notable changes to **QUITS** will be documented in this file.

The project aims to follow Semantic Versioning (SemVer).

## Unreleased

Changes in this section are relative to `v1.2.0`.

## 1.2.0 - 2026-08-03

### Added
- Finite-group / F2[G] group-algebra module (`quits.gf2_util.group`) and
  lifted-product construction module (`quits.gf2_util.lifted_product`),
  following arXiv:2607.28795.
- New public GF(2)/CSS helpers: `gf2_solve_square`, `check_css`, `compute_k`,
  `check_weights`, `qubit_weights`.
- Test suites for the `gf2_util` package: compatibility/golden regression
  tests plus unit tests for the linear-algebra, CSS, group, and lifted-product
  modules.
- `MittenCode` code class (arXiv:2607.28795) with three syndrome-extraction
  strategies: `custom` (depth-12 interleaved mixed X/Z schedule, default),
  `hook_error_free` (depth-24 group-element-layer schedule), and
  `zxcoloration` (depth 18); includes the canonical Theorem-1 logical basis.
  The `hook_error_free` strategy accepts a `seed` that draws a random
  within-section layer order — the search space of the paper's hook-error
  schedule optimization (Appendix H.5); screen seeds with
  `spacetime_matrices`/`estimate_spacetime_distance`.
- The mitten `custom` strategy implements the standard custom-circuit
  interface (`get_custom_schedule_edges`/`get_custom_circuit`, drawing via
  the generic `CustomBuilder`).
- General-purpose schedule checks `interleaving_parity_violations` and
  `verify_stabilizer_flows` in
  `qldpc_code/circuit_construction/schedule_validation.py`, and
  circuit-level distance tools `spacetime_matrices` /
  `estimate_spacetime_distance` in `quits.simulation` — applicable to any
  CSS code, not just mitten codes.
- `ZXColorationBuilder` now accepts optional prescribed per-half layers
  (`layers_Z`/`layers_X`), used by the mitten `hook_error_free` strategy;
  default behavior is unchanged.
- `tests/test_mitten.py` suite covering construction, logicals, all three
  strategy depths, schedule validity, stabilizer flows, detector determinism,
  and an end-to-end sliding-window memory experiment.
- `doc/06C_end_to_end_demo_mitten.ipynb`: end-to-end mitten demo — seed
  optimization of the `hook_error_free` schedule by estimated circuit-level
  distance, compared against the `custom` (depth-12) strategy in depth and
  simulated memory performance.
- Mitten section in `doc/01A_codes_basics.ipynb`, including the published
  code instances of arXiv:2607.28795 and a `[[500, 100]]` example built over
  the non-abelian group `C5 : C20` from explicit permutation generators.

### Changed
- `gf2_util.py` restructured into the `quits/gf2_util/` package
  (`linalg.py`, `css.py`, `group.py`, `lifted_product.py`). All existing
  imports (`from quits.gf2_util import ...`) are preserved unchanged.
- The circulant-lift helpers (`get_circulant_mat`, `lift`, `lift_enc`) moved
  from `qldpc_code/qldpc_util.py` into `gf2_util/lifted_product.py`;
  `qldpc_util.py` remains as a deprecated re-export shim so
  `quits.qldpc_code.qldpc_util` imports keep working. The `QldpcCode`
  deprecation messages now point to `quits.gf2_util`.
- The code-family list in the README now cites each family's paper.

### Removed
- `CardinalNSMergeBuilder`'s `draw_graph` override, so drawing goes through
  the shared `CircuitBuilder` implementation like every other strategy. Its
  strategy-specific `part="direction"` and `part="color"` modes are gone;
  `part` now takes `"node"`, `"all"`, or a layer index, as elsewhere.

## 1.1.0 - 2026-04-27

### Added
- GitHub Actions CI for `pytest` across Python 3.10-3.12.
- PyPI publishing workflow using Trusted Publishing (OIDC).
- `quits.__version__` for runtime version introspection.
- Layout helpers under `src/quits/layout/`, including shared layout abstractions,
  transversal layouts, and toric-layout support for visualization.

### Changed
- Packaging metadata and PyPI README rendering (links and assets).
- Layout documentation now reflects the current module split across
  `src/quits/layout/toric.py`, `toric_common.py`, and `toric_bb.py`.

### Fixed
- README text encoding issues.
