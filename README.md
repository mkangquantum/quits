> 🚀 **New Release Available!**
> **v0.2.0** - base matrices of QLP codes can now be polynomial entries. [Check out the latest release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.2.0)
> **v0.1.0** – important bug is fixed, so please check the release note if you have already been using QUITS.
> [Check out the latest release notes »](https://github.com/mkangquantum/quits/releases/tag/v0.1.0)


# QUITS: A modular Qldpc code circUIT Simulator

QUITS is a modular and flexible circuit-level simulator for quantum low-density parity check (QLDPC) codes. Its design allows users to freely combine LDPC code constructions, syndrome extraction circuits, decoding algorithms, and noise models, enabling comprehensive and customizable studies of the performance of QLDPC codes under circuit-level noise. QUITS supports several leading QLDPC families, including <b>hypergraph product codes, lifted product codes, and balanced product codes</b>. 

Check out [arXiv:2504.02673](https://arxiv.org/abs/2504.02673) for a detailed description of our package. 

QUITS is best used together with the following libraries:
- [Stim](https://github.com/quantumlib/Stim) (fast stabilizer circuit simulator) 
- [LDPC](https://github.com/quantumgizmos/ldpc) (BP-OSD, BP-LSD decoders for QLDPC codes)

See [doc/intro.ipynb](https://github.com/mkangquantum/quits/blob/main/doc/intro.ipynb) to get started!

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
@article{kang2025quantum,
  title={QUITS: A modular Qldpc code circUIT Simulator},
  author={Kang, Mingyu and Lin, Yingjia and Yao, Hanwen and Gökduman, Mert and Meinking, Arianna and Brown, Kenneth R},
  journal={arXiv preprint arXiv:2504.02673},
  year={2025}
}
