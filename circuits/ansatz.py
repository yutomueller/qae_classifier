from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

"""
4.2 Ansatz
P. 13
A variation similar to the linear pattern, the circular pattern, 
adds a CNOT gate to control the first qubit from the last qubit before establishing 
entanglement among adjacent qubits. This creates a circular connection among qubits, 
enhancing entanglement, and is shown as Circuit 3 in Figure 6c.

今は
Circuit 3　環状エンタングルメントのみ
将来的に引数でサーキット変更ができるようにする予定
"""

def build_ansatz(reg, circuit_id=3, reps=20):
    n = len(reg)
    params = ParameterVector('theta', length=reps * n * 2)
    qc = QuantumCircuit(reg)
    p = 0
    for r in range(reps):
        # 1. 全qubitにRY
        for i in range(n):
            qc.ry(params[p], reg[i])
            p += 1
        # 2. CNOTチェーン
        for i in range(n-1):
            qc.cx(reg[i], reg[i+1])
        if circuit_id == 3:
            qc.cx(reg[n - 1], reg[0])
        # 3. (circuit_id==3) RZ層
        if circuit_id == 3:
            for i in range(n):
                qc.rz(params[p], reg[i])
                p += 1
    return qc, params

"""
| アンサッツ  | 任意   | M  | パラ数 | MNIST     | Fashion-MNIST | KMNIST | 平均精度  |
| --------- | ----- | -- | ------ | --------- | ------------- | ------ | --------- |
| Circuit 1 | –     | 20 | 168    | 80.6%     | 67.2%         | 69.6%  | 72.5%     |
| Circuit 2 | –     | 20 | 168    | 87.6%     | 58.8%         | 69.4%  | 71.9%     |
| Circuit 3 | –     | 20 | 168    | 90.4%     | 67.2%         | 65.4%  | 74.3%     |
| Circuit 4 | –     | 6  | 168    | 82.0%     | 69.6%         | 64.8%  | 72.1%     |
| Circuit 5 | RY    | 3  | 216    | 73.6%     | 68.2%         | 66.8%  | 69.5%     |
| Circuit 5 | X     | 11 | 176    | 24.6%     | 25.2%         | 16.4%  | 22.1%     |
| Circuit 6 | RZ    | 5  | 160    | 79.6%     | 49.8%         | 48.4%  | 59.3%     |
| Circuit 6 | RX    | 5  | 160    | 86.2%     | 49.6%         | 61.2%  | 65.7%     |
| Circuit 6 | X     | 10 | 160    | 75.8%     | 67.6%         | 68.0%  | 70.5%     |
| Circuit 7 | RZ    | 7  | 168    | 67.4%     | 50.4%         | 50.4%  | 56.1%     |
| Circuit 7 | RX    | 7  | 168    | 73.6%     | 51.2%         | 49.8%  | 58.2%     |
"""