# 🏦 Bank Customer Churn Prediction ML Pipeline

![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

An end-to-end Machine Learning Operations (MLOps) project that predicts whether a bank customer is likely to churn. The project incorporates model versioning, experiment tracking, pipeline orchestration, and a web-based prediction dashboard.

## 🌟 Key Features

- **ML Pipeline Orchestration**: Uses Data Version Control (**DVC**) to define and manage reproducible data preprocessing, training, and evaluation stages.
- **Experiment Tracking**: Utilizes **MLflow** to track model parameters, metrics (Accuracy, F1 Score), and save the best-performing model artifacts (Logistic Regression, Random Forest, XGBoost).
- **Interactive Dashboard**: A user-friendly **Streamlit** web application for real-time customer churn prediction and risk assessment.
- **Dockerized**: Fully containerized for consistent and easy deployment across environments.

## 📂 Project Structure

```text
bank-customer-churn-mlops/
│
├── app/
│   └── app.py                  # Streamlit web dashboard application
├── data/
│   ├── raw/                    # Original raw dataset
│   └── processed/              # Processed train/test data splits (Pickle/CSV)
├── models/                     # Saved preprocessor and best ML model artifacts
├── reports/                    # Generated metrics (JSON) and plots (Confusion Matrix)
├── src/                        # Machine Learning Pipeline source code
│   ├── preprocess.py           # Data cleaning, scaling, and encoding
│   ├── train.py                # Model training and MLflow tracking
│   └── evaluate.py             # Model evaluation and report generation
│
├── dvc.yaml                    # DVC pipeline stages definition
├── params.yaml                 # Configuration parameters for ML pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Instructions for containerizing the app
└── .dockerignore               # Files omitted from the Docker context
```

## 🚀 Getting Started

### 1. Local Setup

**Clone the repository and install dependencies:**
```bash
git clone <your-repository-url>
cd bank-customer-churn-mlops
pip install -r requirements.txt
```

**Run the Streamlit Dashboard:**
```bash
streamlit run app/app.py
```
*The app will be available at `http://localhost:8501`*

### 2. Docker Setup

Ensure you have Docker Desktop running.

**Build the image:**
```bash
docker build -t bank-customer-churn-mlops .
```

**Run the container:**
```bash
docker run -p 8501:8501 bank-customer-churn-mlops
```
*The app will be available at `http://localhost:8501`*

## 🔄 Running the ML Pipeline

This project uses DVC to manage the ML pipeline defined in `dvc.yaml`. If you wish to retrain the models (e.g., after altering hyperparameters in `params.yaml`), run:

```bash
dvc repro
```

This will automatically:
1. Run `src/preprocess.py` to prepare the data.
2. Run `src/train.py` to train models and track them via MLflow, dynamically accounting for class imbalances.
3. Run `src/evaluate.py` to generate the final confusion matrix and evaluation metrics.

## 📊 Viewing MLflow Experiments

To view the logged experiments, runs, and model metrics, start the MLflow server:

```bash
mlflow ui
```
*Access the MLflow tracking UI at `http://localhost:5000`*

## 📝 Configuration parameters
Model hyperparameters and data splitting logic can be easily modified in `params.yaml` without touching the code.
