import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from qae_classifier import QuantumAutoencoderClassifier

# データロード
data = np.load('../data/digits_preprocessed.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']

# --- QAE分類器インスタンス生成 ---
clf = QuantumAutoencoderClassifier(
    ansatz=3,      # 今は３のみ
    reps=10,       # 回路の深さ number of repetitions in each ansatz.
    shots=512,
    epochs=2000,    # 学習エポック（短めでサンプル）
    optimizer='COBYLA'
)

# --- モデル学習 ---
clf.train(X_train, y_train)

for _ in range(3):
    # --- 推論 ---
    y_pred = clf.predict(X_test)

    # --- 精度評価 ---
    from sklearn.metrics import accuracy_score, classification_report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))




