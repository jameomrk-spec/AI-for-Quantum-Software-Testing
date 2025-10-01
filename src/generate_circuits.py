from qiskit import QuantumCircuit
from qiskit.qasm3 import dump

def generate_circuits():
    # ✅ Clean circuit
    qc_clean = QuantumCircuit(2)
    qc_clean.h(0)
    qc_clean.cx(0, 1)
    qc_clean.measure_all()

    with open("data/clean.qasm", "w") as f:
        dump(qc_clean, f)

    # ❌ Buggy circuit (missing measurement)
    qc_buggy = QuantumCircuit(2)
    qc_buggy.h(0)
    qc_buggy.cx(0, 1)

    with open("data/buggy.qasm", "w") as f:
        dump(qc_buggy, f)

if __name__ == "__main__":
    generate_circuits()
    print("Saved circuits in data/")
