> 🚀 **New Release Available!**
>**v0.4.0** - Canonical logical codewords of HGP code fixed - please check the release note if you have already been using QUITS. `gf2_util.py` and test functions added. [Check out the latest release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.4.0)
> 
>**v0.3.0** - Logical codewords of HGP and BPC codes are now given in the "canonical" form. [Check out the latest release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.3.0)
> 
> **v0.2.0** - base matrices of QLP codes can now be polynomial entries. [Check out the latest release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.2.0)
> 
> **v0.1.0** – important bug is fixed, so please check the release note if you have already been using QUITS.
> [Check out the release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.1.0)


# QUITS: A modular Qldpc code circUIT Simulator

QUITS is a modular and flexible circuit-level simulator for quantum low-density parity check (QLDPC) codes. Its design allows users to freely combine LDPC code constructions, syndrome extraction circuits, decoding algorithms, and noise models, enabling comprehensive and customizable studies of the performance of QLDPC codes under circuit-level noise. QUITS supports several leading QLDPC families, including 
- HyperGraph Product (HGP) codes 
- Quasi-cyclic Lifted Product (QLP) codes, and 
- Balanced Product Cyclic (BPC) codes. 

Check out [arXiv:2504.02673](https://arxiv.org/abs/2504.02673) for a detailed description of our package. 

QUITS is best used together with the following libraries:
- [Stim](https://github.com/quantumlib/Stim) (fast stabilizer circuit simulator) 
- [LDPC](https://github.com/quantumgizmos/ldpc) (BP-OSD, BP-LSD decoders for QLDPC codes)

See [doc/intro.ipynb](https://github.com/mkangquantum/quits/blob/main/doc/intro.ipynb) to get started!

Since the release of QUITS, we acknowledge the feedback and suggestions from Ryan Tiew, Josias Old, and qodesign that helped improve the package. If you’re working on QLDPC codes, decoders, or noise modeling, it'd be great if you could try QUITS, file issues, or contribute features. Let’s build better tools for scalable, fault-tolerant quantum computing together ⚛️

## Installation

To install this package from GitHub, use the following steps:

1. **Clone the repository:**
   ```
   git clone https://github.com/mkangquantum/quits.git
   ```
   
2. **Navigate to the repository**

3. **Run installation command**
   ```
   pip install -e .
   ```

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
