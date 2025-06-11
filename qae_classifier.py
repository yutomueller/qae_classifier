import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from circuits.encoder import build_encoder
from circuits.ansatz import build_ansatz
from circuits.label_ref import build_label_ref

class QuantumAutoencoderClassifier:
    def __init__(self, ansatz=3, reps=20, label_qubits=None, shots=256, optimizer='COBYLA', epochs=500):
        self.ansatz = ansatz
        self.reps = reps # number of repetitions in each ansatz.
        self.label_qubits = label_qubits  # Noneなら自動
        self.shots = shots
        self.optimizer = optimizer
        self.epochs = epochs

        self.latent_qubits = None
        self.n_features = None
        self.n_classes = None      # ラベル y の最大値 +1 により自動計算。例：y = [0, 1, 2] → self.n_classes = 3

        self.templates = None      # 学習用transpiled回路（パラメータ未設定状態）
        self.theta_params = None   # ParameterVector
        self.trained_params = None # 学習済みパラメータ

    def train(self, X, y):
        """
        QAE論文準拠:
        - 各サンプルごとにパラメータ未バインドのQuantumCircuitとParameterVectorを保存
        - テンプレごとにtranspile済み回路・theta_paramsをペアで記録
        - cost_fnでその都度 assign_parameters して評価
        """
        self.n_features = X.shape[1]
        self.n_classes = int(np.max(y)) + 1  # 0,1,2,3...
        self.label_qubits = self.label_qubits or int(np.ceil(np.log2(self.n_classes)))
        self.latent_qubits = int(np.log2(self.n_features)) - self.label_qubits
        if self.latent_qubits <= 0:
            raise ValueError("latent_qubits <= 0。画像サイズかlabel_qubits設定を見直してください。")

        backend = AerSimulator(method='statevector')
        print("Pre-compiling transpiled templates...")
        self.templates = []
        for x, cls in zip(X, y):
            qc, theta_params = self._build_train_circuit(x, cls)
            qc_t = transpile(qc, backend)
            self.templates.append({"transpiled_circ": qc_t, "theta_params": theta_params})

        n_param = len(self.templates[0]["theta_params"])
        init_params = np.random.uniform(0, 2 * np.pi, n_param)

        def cost_fn(params):
            total = 0
            for tpl in self.templates:
                param_dict = {tpl["theta_params"][i]: params[i] for i in range(n_param)}
                bound_circ = tpl["transpiled_circ"].assign_parameters(param_dict)
                job = backend.run(bound_circ, shots=self.shots).result()
                counts = job.get_counts()
                prob1 = counts.get('1', 0) / self.shots
                total += prob1
            return total / len(self.templates)

        print("Starting optimization...")
        res = minimize(
            cost_fn, init_params,
            method=self.optimizer,
            options={'maxiter': self.epochs, 'disp': True}
        )
        self.trained_params = res.x
        print("Training complete.")
        return self

    def predict(self, X):
        backend = AerSimulator(method='statevector')
        y_pred = []
        theta_params = self.templates[0]['theta_params']
        n_param = len(theta_params)
        for x in X:
            qc, pred_params = self._build_predict_circuit(x)
            param_dict = {pred_params[i]: self.trained_params[i] for i in range(n_param)}
            bound_circ = qc.assign_parameters(param_dict)
            qc_t = transpile(bound_circ, backend)
            job = backend.run(qc_t, shots=self.shots).result()
            counts = job.get_counts()
            # 多クラス（例：3ビットなら '010'）の最大頻度のbit列を使う
            y_pred.append(int(max(counts, key=counts.get), 2))
        return np.array(y_pred)

    def predict_proba(self, X):
        backend = AerSimulator(method='statevector')
        all_probs = []
        theta_params = self.templates[0]['theta_params']
        n_param = len(theta_params)
        for x in X:
            qc, pred_params = self._build_predict_circuit(x)
            param_dict = {pred_params[i]: self.trained_params[i] for i in range(n_param)}
            bound_circ = qc.assign_parameters(param_dict)
            qc_t = transpile(bound_circ, backend)
            job = backend.run(qc_t, shots=self.shots).result()
            counts = job.get_counts()

            # クラスごとの出現確率を格納
            probs = [0.0 for _ in range(self.n_classes)]
            for bitstr, count in counts.items():
                idx = int(bitstr, 2)
                if idx < self.n_classes:
                    probs[idx] += count / self.shots
            # 念のため正規化（合計1.0になるはずだが、端数誤差を防ぐため）
            total = sum(probs)
            probs = [p / total for p in probs]
            all_probs.append(probs)
        return np.array(all_probs)

    def _build_train_circuit(self, x, cls):
        """
        Fig. 3: Quantum autoencoder circuit for image-classification training. 
                Class information is encoded in VL(y)
        """
        reg_a = QuantumRegister(self.latent_qubits, 'latent')
        reg_b = QuantumRegister(self.label_qubits, 'label')
        reg_bref = QuantumRegister(self.label_qubits, 'ref')
        anc = QuantumRegister(1, 'anc')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(reg_a, reg_b, reg_bref, anc, c)
        # 厳密正規化
        x = x.astype(np.complex128)
        norm = np.linalg.norm(x)
        if not np.isclose(norm, 1.0, atol=1e-10):
            x = x / norm
        if len(x) != 2 ** (self.latent_qubits + self.label_qubits):
            raise ValueError(
                f"Input vector length ({len(x)}) != 2^{self.latent_qubits + self.label_qubits} qubits."
            )
        """
        Fig 3
        """ 
        qc = qc.compose(build_encoder(x, list(reg_a) + list(reg_b)), qubits=list(reg_a) + list(reg_b)) #V(x)
        ansatz_circ, theta_params = build_ansatz(list(reg_a) + list(reg_b), self.ansatz, self.reps) # U(theta)
        qc = qc.compose(ansatz_circ, qubits=list(reg_a) + list(reg_b))
        qc = qc.compose(build_label_ref(cls, reg_bref), qubits=reg_bref) # VL(y)
        self._swap_test(qc, reg_b, reg_bref, anc[0])
        qc.measure(anc[0], c[0])
        return qc, theta_params

    def _build_predict_circuit(self, x):
        reg_a = QuantumRegister(self.latent_qubits, 'latent')
        reg_b = QuantumRegister(self.label_qubits, 'label')
        c = ClassicalRegister(self.label_qubits, 'clabel')
        qc = QuantumCircuit(reg_a, reg_b, c)
        x = x.astype(np.complex128)
        norm = np.linalg.norm(x)
        if not np.isclose(norm, 1.0, atol=1e-10):
            x = x / norm
        """
        Figure 5 shows the circuit used for classifying images using the trained θ. The
        parameters of V(x) are set using the information from the test image to be classified
        and the trained parameters are applied to U(θ). The circuit is then measured.
        """
        qc = qc.compose(build_encoder(x, list(reg_a) + list(reg_b)), qubits=list(reg_a) + list(reg_b)) #V(x)
        ansatz_circ, theta_params = build_ansatz(list(reg_a) + list(reg_b), self.ansatz, self.reps) #U(theta)
        qc = qc.compose(ansatz_circ, qubits=list(reg_a) + list(reg_b))
        # Bレジスタのみ測定
        for i in range(self.label_qubits):
            qc.measure(reg_b[i], c[i])
        return qc, theta_params

    @staticmethod
    def _swap_test(qc, reg_b, reg_bref, anc):
        """
        スワップテスト回路
        Fig. 2: Quantum autoencoder structure after conducting the swap test
        """
        qc.h(anc)
        for qb, qbr in zip(reg_b, reg_bref):
            qc.cswap(anc, qb, qbr)
        qc.h(anc)





