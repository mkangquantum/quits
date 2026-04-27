# QUITS - Modular QLDPC Code Circuit Simulator

> CLAUDE.md and AGENTS.md are identical. Keep them in sync.

## Overview

QUITS (Qldpc code circUIT Simulator) is a modular, circuit-level simulator for quantum low-density parity-check (QLDPC) codes. Users can independently choose and combine code constructions, circuit strategies, decoders, noise models, and layout helpers.

Paper: [arXiv:2504.02673](https://arxiv.org/abs/2504.02673) / [Quantum 9, 1931 (2025)](https://doi.org/10.22331/q-2025-12-05-1931)

## Package Architecture

```text
src/quits/
|-- __init__.py             # Package entrypoint re-exporting public API
|-- api.py                  # Public API facade
|-- circuit.py              # Stim circuit generation
|-- simulation.py           # End-to-end simulation pipeline
|-- gf2_util.py             # GF(2) linear algebra helpers
|-- ldpc_util.py            # Classical LDPC matrix generation
|-- qldpc_code/             # QLDPC code families and utilities
|   |-- __init__.py
|   |-- base.py             # Base code class
|   |-- hgp.py              # Hypergraph Product codes
|   |-- bb.py               # Bivariate Bicycle codes
|   |-- bpc.py              # Balanced Product Cyclic codes
|   |-- lcs.py              # Lift-Connected Surface codes
|   |-- qlp.py              # Quasi-cyclic Lifted Product codes
|   |-- qldpc_util.py       # Shared QLDPC utilities
|   `-- circuit_construction/
|       |-- __init__.py
|       |-- base.py                  # Base circuit builder
|       |-- cardinal.py              # Cardinal scheduling strategy
|       |-- cardinalNSmerge.py       # Cardinal variant with NS merge scheduling
|       |-- custom.py                # Custom BB-specific strategy
|       |-- zxcoloration.py          # ZX-coloration strategy
|       |-- edge_coloration.py       # Tanner-graph edge coloring
|       `-- circuit_build_options.py # Build option dataclass
|-- decoder/
|   |-- __init__.py
|   |-- base.py             # Base decoder interface
|   |-- bposd.py            # BP-OSD decoder
|   |-- bplsd.py            # BP-LSD decoder
|   `-- sliding_window.py   # Sliding-window decoder for circuit-level decoding
|-- noise/
|   |-- __init__.py
|   `-- error_model.py      # Structured noise model (idle/gate/SPAM rates)
`-- layout/
    |-- __init__.py
    |-- base.py             # Layout and layout-mapping abstractions
    |-- transversal.py      # Generic row-based Tanner-graph layouts
    |-- toric.py            # Public toric layout factory
    |-- toric_common.py     # Shared toric embedding machinery
    |-- toric_bb.py         # BB toric layouts
```

## Supported Code Families

| Family | Key | Circuit Strategies |
|--------|-----|--------------------|
| Hypergraph Product | `hgp` | `zxcoloration`, `cardinal`, `cardinalNSmerge` |
| Quasi-cyclic Lifted Product | `qlp` | `zxcoloration`, `cardinal`, `cardinalNSmerge` |
| Balanced Product Cyclic | `bpc` | `zxcoloration`, `cardinal`, `cardinalNSmerge` |
| Lift-Connected Surface | `lcs` | `zxcoloration`, `cardinal`, `cardinalNSmerge` |
| Bivariate Bicycle | `bb` | `zxcoloration`, `custom` |
| Bring-your-own matrix | N/A | `zxcoloration` |

## Key Dependencies

- `stim` - stabilizer circuit simulation
- `ldpc` - BP-OSD / BP-LSD decoding
- `numpy`, `scipy` - linear algebra
- `networkx` - Tanner-graph operations
- `matplotlib` - visualization

## Repo Layout

- `src/quits/` - main package (`python >= 3.10`, package version `1.0.0`)
- `tests/` - pytest suite covering codes, circuits, decoders, edge coloration, sliding window, and `cardinalNSmerge`
- `doc/` - Jupyter notebooks from `00_getting_started.ipynb` through end-to-end demos (`06A`, `06B`)
- `examples/` - standalone scripts such as circuit-distance search
- `parity_check_matrices/` - precomputed LDPC / QLDPC matrices
