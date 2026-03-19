# 🏦 Bank Customer Churn Prediction ML Pipeline

![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Frontend](https://img.shields.io/badge/HTML5/CSS3/JS-Premium%20UI-orange)

An end-to-end Machine Learning Operations (MLOps) project that predicts whether a bank customer is likely to churn. This project features a enterprise-grade architecture with a FastAPI backend and a custom, high-performance HTML/CSS/JS dashboard.

## 🌟 Key Features

- **Production API Layer**: Powered by **FastAPI** for high-performance, asynchronous model serving.
- **Premium Dark Dashboard**: A custom-built **HTML/CSS/JS** frontend with glassmorphism, animated risk charts, and real-time response handling.
- **ML Pipeline Orchestration**: Uses **DVC** to manage reproducible data preprocessing, training, and evaluation stages.
- **Experiment Tracking**: Utilizes **MLflow** to track model parameters, metrics (Accuracy, F1 Score), and save the best-performing model artifacts.
- **Dockerized**: Fully containerized for consistent and easy deployment across environments.

## 📂 Project Structure

```text
bank-customer-churn-mlops/
├── backend/            # FastAPI server and prediction logic (main.py)
├── frontend/           # Premium web dashboard (HTML, CSS, JS)
├── data/               # Raw and processed datasets (DVC tracked)
├── models/             # Saved model and preprocessor artifacts
├── src/                # ML Pipeline source code (preprocess, train, evaluate)
├── reports/            # Generated metrics and confusion matrix plots
├── dvc.yaml            # DVC pipeline stages definition
├── params.yaml         # Centralized configuration parameters
├── requirements.txt    # Python dependencies
└── Dockerfile          # Container instructions (Port 8000)
```

## 🚀 Getting Started

### 1. Local Setup (Anaconda)

**Create and activate the environment:**
```bash
conda create -n churn-mlops python=3.9 -y
conda activate churn-mlops
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Production Dashboard:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Access the dashboard at `http://localhost:8000`*

### 2. Docker Setup

**Build the image:**
```bash
docker build -t bank-customer-churn-mlops .
```

**Run the container:**
```bash
docker run -p 8000:8000 bank-customer-churn-mlops
```
*Access the dashboard at `http://localhost:8000`*

## 🔄 Running the ML Pipeline

This project uses DVC to manage the ML pipeline. To retrain models after modifying `params.yaml`:

```bash
dvc repro
```

This will automatically:
1. Run `src/preprocess.py` to prepare data.
2. Run `src/train.py` to train models and track them via MLflow.
3. Run `src/evaluate.py` to generate final reports.

## 📊 Viewing MLflow Experiments

To view the logged experiments and model metrics:

```bash
mlflow ui
```
*Access the MLflow tracking UI at `http://localhost:5000`*

## 📝 Configuration parameters
Model hyperparameters and data splitting logic can be easily modified in `params.yaml` without touching the code.
