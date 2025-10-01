import os
import pandas as pd
from qiskit.qasm3 import loads

def extract_features_from_qasm(qasm_file):
    """Extract simple features from a QASM file."""
    with open(qasm_file, "r") as f:
        circuit = loads(f.read())

    # Features: number of qubits, number of gates, depth, has_measure
    num_qubits = circuit.num_qubits
    num_gates = circuit.size()
    depth = circuit.depth()
    has_measure = any(op.operation.name == "measure" for op in circuit)

    return {
        "num_qubits": num_qubits,
        "num_gates": num_gates,
        "depth": depth,
        "has_measure": int(has_measure),
        "label": 1 if "clean" in qasm_file else 0,  # clean=1, buggy=0
    }

def build_dataset(data_folder="data", output_csv="data/features.csv"):
    rows = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".qasm"):
            filepath = os.path.join(data_folder, filename)
            features = extract_features_from_qasm(filepath)
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved dataset to {output_csv}")
    return df

if __name__ == "__main__":
    build_dataset()
