from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def build_label_ref(y, reg):
    """y: int label (0–3), reg: 3-qubit register"""
    qc = QuantumCircuit(reg)
    bits = [int(b) for b in format(y, f'0{len(reg)}b')]
    for i, b in enumerate(reversed(bits)):
        if b == 1:
            qc.rx(np.pi, reg[i])  # bit-flip
    return qc
