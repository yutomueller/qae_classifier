import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from qae_classifier import QuantumAutoencoderClassifier  # あなたの実装に合わせて修正

# 1. 数値データ生成
X, y = make_classification(
    n_samples=150,
    n_features=8,       # ← 2のべき乗（QAE要件を満たす）
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    class_sep=1.5,
    random_state=42
)

# 2. データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. QAE 分類器の初期化
clf = QuantumAutoencoderClassifier(
    ansatz=3,
    reps=8,
    shots=128,
    epochs=500,       # デバッグ用に短くしておく
    optimizer='COBYLA'
)

# 4. モデル訓練
clf.train(X_train, y_train)

# 5. 予測と評価（複数回評価してばらつきを見るのもアリ）
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

