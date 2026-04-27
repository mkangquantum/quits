# Changelog

All notable changes to **QUITS** will be documented in this file.

The project aims to follow Semantic Versioning (SemVer).

## Unreleased

Changes in this section are relative to `v1.0.0`.

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
