# Changelog

All notable changes to **QUITS** will be documented in this file.

The project aims to follow Semantic Versioning (SemVer).

## Unreleased

Changes in this section are relative to `v1.1.0`.

### Added
- Finite-group / F2[G] group-algebra module (`quits.gf2_util.group`) and
  lifted-product / mitten-code construction module
  (`quits.gf2_util.lifted_product`), following arXiv:2607.28795.
- New public GF(2)/CSS helpers: `gf2_solve_square`, `check_css`, `compute_k`,
  `check_weights`, `qubit_weights`.
- Test suites for the `gf2_util` package: compatibility/golden regression
  tests plus unit tests for the linear-algebra, CSS, group, and lifted-product
  modules.

### Changed
- `gf2_util.py` restructured into the `quits/gf2_util/` package
  (`linalg.py`, `css.py`, `group.py`, `lifted_product.py`). All existing
  imports (`from quits.gf2_util import ...`) are preserved unchanged.
- The circulant-lift helpers (`get_circulant_mat`, `lift`, `lift_enc`) moved
  from `qldpc_code/qldpc_util.py` into `gf2_util/lifted_product.py`;
  `qldpc_util.py` remains as a deprecated re-export shim so
  `quits.qldpc_code.qldpc_util` imports keep working. The `QldpcCode`
  deprecation messages now point to `quits.gf2_util`.

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
