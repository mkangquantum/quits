# QUITS: A modular Qldpc code circUIT Simulator

QUITS is a modular and flexible circuit-level simulator for quantum low-density parity check (QLDPC) codes. Its design allows users to freely combine LDPC code constructions, syndrome extraction circuits, decoding algorithms, and noise models, enabling comprehensive and customizable studies of the performance of QLDPC codes under circuit-level noise. QUITS supports several leading QLDPC families, including hypergraph product codes, lifted product codes, and balanced product codes.

QUITS is best used together with the following libraries:
- Stim (fast stabilizer circuit simulator) https://github.com/quantumlib/Stim
- LDPC (provides decoders for QLDPC codes) https://github.com/quantumgizmos/ldpc

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