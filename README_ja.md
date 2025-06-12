# 🧠 QuantumAutoencoderClassifier

量子オートエンコーダを用いた教師あり分類器  
Supervised image classification using quantum autoencoders  
**Based on:** [Quantum Autoencoders for Image Classification](https://arxiv.org/abs/2502.15254)

---

## 🌟 特徴 / Features

- 量子オートエンコーダによる教師あり分類
- 論文に準拠した回路構成（V(x), U(θ), label ref, Swap test）
- `Qiskit 2.0.2 + Aer 0.17.0` に対応
- `train`, `predict` メソッドを実装

---

## 📘 論文との対応表 / Algorithm Mapping to Paper

| 論文記述 / Paper Section         | 実装内容 / Functionality                     | 実装場所 / Implemented in                   |
|----------------------------------|-----------------------------------------------|---------------------------------------------|
| **Fig. 5**                       | 分類回路構成: V(x) + U(θ) + 測定              | `_build_predict_circuit()`                  |
| **Fig. 6**                       | アンザッツ U(θ) 現在Circuit 3のみ          | `build_ansatz()`（in `circuits/ansatz.py`） |
| **Sec 4.1**                   | 振幅埋め込み: V(x)                            | `build_encoder()`（in `circuits/encoder.py`）|
| **Fig. 3**                       | ラベル参照状態 V_L(y)                         | `build_label_ref()`（in `circuits/label_ref.py`）|
| **Ansatz repetition M**         | U(θ)の繰り返し回数指定（M）                   | `reps` 引数で制御                           |

---

## 🧪 動作条件 / Requirements

- Python ≥ 3.9
- `qiskit == 2.0.2`
- `qiskit-aer == 0.17.0`
- 実機・Simulator対応（推奨: `AerSimulator(method="statevector")`）

## 📊 入力要件 / Input Requirements

🔢 入力 X の形式について
- 各サンプル x は NumPy 配列で、複素数の振幅ベクトルとして扱われます。
- 長さは 2ⁿ（2のべき乗） にする必要があります。
- → n = latent_qubits + label_qubits

- x は L2ノルム = 1 に正規化されたベクトルである必要があります。

```python
x = x / np.linalg.norm(x)
```
- 実数でも複素数でも動作します（np.complex128 に自動変換されます）

## 🧪 ラベル y の形式
- y は int 型のクラスラベル配列（例：[0, 1, 0, 1, ...]）
- クラス数は yの最大値+1 個として、自動で処理されます

---

## 🗂️ ディレクトリ構成 / File Structure

```
qae_image_classification/
├── qae_classifier.py              # メインクラス
├── circuits/
│   ├── encoder.py                 # build_encoder
│   ├── ansatz.py                  # build_ansatz
│   └── label_ref.py               # build_label_ref
├── data 
│   ├── preprocess_digits.py       # データ生成
│   └── digits_preprocesser.npz    # exampleで使うデータ
├── example
│    └── example.py                 # 学習 & 推論例
└── README.md                      # ← 本ファイル
```

---

## 🔗 論文リンク / Paper Link

- **Quantum Autoencoders for Image Classification**  
  [arXiv:2502.15254](https://arxiv.org/abs/2502.15254)
