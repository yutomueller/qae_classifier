# 🧠 QuantumAutoencoderClassifier

Supervised image classification using quantum autoencoders  
**Based on:** [Quantum Autoencoders for Image Classification](https://arxiv.org/abs/2502.15254)

---

## 🌟 Features

- Supervised image classification using quantum autoencoders
- Circuit architecture faithfully follows the paper: V(x), U(θ), label reference, and Swap test
- Compatible with `Qiskit 2.0.2 + Aer 0.17.0`
- Implements `train` and `predict` methods

---

## 📘 Algorithm Mapping to Paper

| Paper Section         | Functionality                               | Implemented in                             |
|-----------------------|----------------------------------------------|---------------------------------------------|
| **Fig. 5**            | Classification circuit: V(x) + U(θ) + measurement | `_build_predict_circuit()`              |
| **Fig. 6**            | Ansatz U(θ), currently supports Circuit 3    | `build_ansatz()` in `circuits/ansatz.py`   |
| **Sec. 4.1**          | Amplitude embedding: V(x)                    | `build_encoder()` in `circuits/encoder.py` |
| **Fig. 3**            | Label reference state V_L(y)                 | `build_label_ref()` in `circuits/label_ref.py` |
| **Ansatz repetition M** | Repetition count M of U(θ)                | Controlled via `reps` argument              |

---

## 🧪 Requirements

- Python ≥ 3.9
- `qiskit == 2.0.2`
- `qiskit-aer == 0.17.0`
- Works on both quantum simulators and real hardware (recommended: `AerSimulator(method="statevector")`)

---

## 📊 Input Requirements

🔢 Format of input X:
- Each sample `x` should be a NumPy array treated as a complex amplitude vector
- Its length must be a power of two:  
  → `len(x) = 2^(latent_qubits + label_qubits)`
- Input vectors must be normalized to L2-norm = 1

```python
x = x / np.linalg.norm(x)
```

- Real or complex vectors are accepted (automatically cast to `np.complex128`)

### 🧪 Format of label y

- `y` is an array of integer class labels (e.g., [0, 1, 0, 1, ...])
- The number of classes is automatically inferred as `max(y) + 1`

---

## 🗂️ File Structure

```
qae_image_classification/
├── qae_classifier.py              # Main classifier class
├── circuits/
│   ├── encoder.py                 # build_encoder
│   ├── ansatz.py                  # build_ansatz
│   └── label_ref.py               # build_label_ref
├── data 
│   ├── preprocess_digits.py       # Preprocessing code
│   └── digits_preprocesser.npz    # Sample dataset for demo
├── example
│    └── example.py                # Training & inference example
└── README.md                      # ← This file
```

---

## 🔗 Paper Reference

- **Quantum Autoencoders for Image Classification**  
  [arXiv:2502.15254](https://arxiv.org/abs/2502.15254)

---
