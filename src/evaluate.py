import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    X_test, y_test = joblib.load("data/processed/test.pkl")
    return model, X_test, y_test


def evaluate():
    model, X_test, y_test = load_artifacts()

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("\nClassification Report:\n")
    print(report)

    # Save metrics
    os.makedirs("reports", exist_ok=True)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    print("\nEvaluation artifacts saved in reports/")


if __name__ == "__main__":
    evaluate()