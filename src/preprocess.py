import pandas as pd
import os
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["data"]["test_size"]
random_state = params["data"]["random_state"]


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Drop unnecessary columns
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Define column types
    categorical_cols = ["Geography", "Gender"]
    numerical_cols = [
        "CreditScore", "Age", "Tenure",
        "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember",
        "EstimatedSalary"
    ]

    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Fit and transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


def save_data(X_train, X_test, y_train, y_test, preprocessor):

    os.makedirs("data/processed", exist_ok=True)

    joblib.dump((X_train, y_train), "data/processed/train.pkl")
    joblib.dump((X_test, y_test), "data/processed/test.pkl")

    joblib.dump(preprocessor, "models/preprocessor.pkl")


if __name__ == "__main__":

    df = load_data("data/raw/Churn_Modelling.csv")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    save_data(X_train, X_test, y_train, y_test, preprocessor)

    print("Preprocessing completed successfully.")