# QUITS: A Modular QLDPC Code circUIT Simulator

QUITS is a modular and flexible circuit-level simulator for quantum low-density parity-check (QLDPC) codes. It is designed to combine code construction, circuit generation, decoding, and noise modeling in a composable workflow.

## Modular Architecture

QUITS is organized into clear modules:

- `quits.qldpc_code`: QLDPC code families and code objects.
- `quits.qldpc_code.circuit_construction`: circuit-construction strategies and options.
- `quits.decoder`: sliding-window and related decoding routines.
- `quits.noise.ErrorModel`: structured noise-model configuration for circuit generation.

Supported code families include:

- Hypergraph Product (HGP) codes
- Quasi-cyclic Lifted Product (QLP) codes
- Balanced Product Cyclic (BPC) codes
- Lift-connected Surface-like (LSC) codes
- Bivariate Bicycle (BB) codes
- **Any code**, given parity check matrices

For background on QUITS, see [arXiv:2504.02673](https://arxiv.org/abs/2504.02673).

## Circuit Construction Strategies

| Code family | `zxcoloration` | `cardinal` | `custom` |
| --- | --- | --- | --- |
| HGP | yes | yes | no |
| QLP | yes | yes | no |
| BPC | yes | yes | no |
| LSC | yes | yes | no |
| BB | yes | no | yes |
| Any | yes | no | no |

- `zxcoloration` is available for all QLDPC codes.
- `cardinal` is available for HGP, QLP, BPC, and LSC.
- `custom` is available for BB code construction.

## Installation

Conda-first workflow:

```bash
conda create -n quits python=3.12 -y
conda activate quits
pip install quits
```

For source/development installs from this repository:

```bash
pip install -e .
```

## Quick Start Docs

- [doc/00_getting_started.ipynb](doc/00_getting_started.ipynb)

## License

This project is licensed under the MIT License.

## How to Cite Our Work

If you use our work in your research, please cite it using the following reference:

```bibtex
@article{Kang2025quitsmodularqldpc,
  doi = {10.22331/q-2025-12-05-1931},
  url = {https://doi.org/10.22331/q-2025-12-05-1931},
  title = {{QUITS}: {A} modular {Q}ldpc code circ{UIT} {S}imulator},
  author = {Kang, Mingyu and Lin, Yingjia and Yao, Hanwen and G{\"{o}}kduman, Mert and Meinking, Arianna and Brown, Kenneth R.},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {9},
  pages = {1931},
  month = dec,
  year = {2025}
}
```
