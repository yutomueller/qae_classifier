import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

"""
4.1 Experimental settings
P.10 amplitude encoding is used as the image encoder V(x) 
"""


def build_encoder(x: np.ndarray, reg: QuantumRegister) -> QuantumCircuit:
    qc = QuantumCircuit(reg)
    x = x.astype(np.complex128)
    norm = np.linalg.norm(x)
    if not np.isclose(norm, 1.0, atol=1e-10):
        x = x / norm
    qc.initialize(x, reg)
    return qc

"""
~P.10
Various effective methods for encoding classical image data into quantum states
have been proposed, such as basis encoding, angle encoding, novel enhanced quantum
representation (NEQR), and flexible representation of quantum images (FRQI) (Rath
and Date, 2024; Zhang et al., 2013; Le et al., 2011). However, these methods require a
greater number of qubits compared to amplitude encoding. While FRQI is relatively
efficient in terms of qubit usage, its practical implementation remains complex due to
the need to execute quantum circuits during the encoding process. Amplitude encoding
is sufficient to represent the information in the simple image datasets used in this
experiment.
"""