import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import joblib

def analyze_results(features_csv="data/features.csv", model_file="results/model.joblib"):
    # Load data and model
    df = pd.read_csv(features_csv)
    X = df.drop("label", axis=1)
    y_true = df["label"]

    model = joblib.load(model_file)
    y_pred = model.predict(X)

    # 🔹 Print classification report
    print("📊 Classification Report:\n")
    print(classification_report(y_true, y_pred))

    # 🔹 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Buggy (0)", "Clean (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    print("✅ Saved confusion matrix to results/confusion_matrix.png")

    # 🔹 Feature Importance (from Random Forest)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = X.columns
        plt.bar(features, importance)
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        plt.xticks(rotation=30)
        plt.savefig("results/feature_importance.png")
        plt.close()
        print("✅ Saved feature importance to results/feature_importance.png")

if __name__ == "__main__":
    analyze_results()
