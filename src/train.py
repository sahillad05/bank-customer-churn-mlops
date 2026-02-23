import joblib
import yaml
import os
import mlflow
import mlflow.sklearn

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

random_state = params["data"]["random_state"]


def load_data():
    X_train, y_train = joblib.load("data/processed/train.pkl")
    X_test, y_test = joblib.load("data/processed/test.pkl")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    return acc, f1


def train():

    X_train, X_test, y_train, y_test = load_data()

    # 🔥 Dynamically calculate class imbalance weight
    counter = Counter(y_train)
    neg = counter[0]
    pos = counter[1]
    scale_pos_weight = neg / pos

    print(f"\nClass Distribution → Negative: {neg}, Positive: {pos}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}\n")

    mlflow.set_experiment("churn-mlops-experiment")

    best_f1 = 0
    best_model = None
    best_model_name = ""

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=params["models"]["logistic_regression"]["max_iter"],
            random_state=random_state
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=params["models"]["random_forest"]["n_estimators"],
            max_depth=params["models"]["random_forest"]["max_depth"],
            random_state=random_state
        ),

        "XGBoost": XGBClassifier(
            n_estimators=params["models"]["xgboost"]["n_estimators"],
            max_depth=params["models"]["xgboost"]["max_depth"],
            learning_rate=params["models"]["xgboost"]["learning_rate"],
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            acc, f1 = evaluate_model(model, X_test, y_test)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\n🏆 Best Model: {best_model_name} with F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    train()