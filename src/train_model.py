import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(csv_file="data/features.csv", model_file="results/model.joblib"):
    # Load dataset
    df = pd.read_csv(csv_file)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_file)
    print(f"💾 Saved model to {model_file}")

if __name__ == "__main__":
    train_model()
