import os
import joblib
import pandas as pd
from qiskit.qasm3 import loads
from src.feature_extraction import extract_features_from_qasm

def evaluate_model(model_file="results/model.joblib", data_folder="data"):
    # Load trained model
    model = joblib.load(model_file)
    print(f"✅ Loaded model from {model_file}")

    rows = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".qasm"):
            filepath = os.path.join(data_folder, filename)
            features = extract_features_from_qasm(filepath)
            rows.append((filename, features))

    # Convert to DataFrame
    df = pd.DataFrame([f for _, f in rows])
    X = df.drop("label", axis=1)
    y_true = df["label"]
    y_pred = model.predict(X)

    for (fname, _), pred, true in zip(rows, y_pred, y_true):
        status = "✅ Correct" if pred == true else "❌ Wrong"
        print(f"{fname} → Predicted: {pred}, Actual: {true} → {status}")

if __name__ == "__main__":
    evaluate_model()
