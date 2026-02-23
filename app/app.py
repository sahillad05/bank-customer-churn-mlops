import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a6fa5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f4f8;
        border-left: 4px solid #1a3c5e;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .churn {
        background-color: #ffe4e4;
        border: 2px solid #e74c3c;
        color: #c0392b;
    }
    .no-churn {
        background-color: #e4f8e4;
        border: 2px solid #27ae60;
        color: #1e8449;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a3c5e;
        border-bottom: 2px solid #4a6fa5;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "best_model.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_preprocessor():
    preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"
    try:
        preprocessor = joblib.load(preprocessor_path)
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None


def load_metrics():
    metrics_path = BASE_DIR / "reports" / "metrics.json"
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_confusion_matrix():
    cm_path = BASE_DIR / "reports" / "confusion_matrix.png"
    if cm_path.exists():
        return str(cm_path)
    return None


model = load_model()
preprocessor = load_preprocessor()

st.markdown('<div class="main-header">🏦 Bank Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Customer Retention Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### 📋 About This Tool")
    st.markdown(
        "This dashboard uses a machine learning model trained on bank customer data "
        "to predict the likelihood of a customer churning. "
        "Fill in the customer details on the right to get an instant prediction."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    metrics = load_metrics()
    if metrics:
        for key, value in metrics.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                st.metric(label=label, value=f"{value:.4f}")
            else:
                st.metric(label=label, value=value)
    else:
        st.info("Metrics file not found.")

    st.markdown("---")
    st.markdown("### 🖼 Confusion Matrix")
    cm_path = load_confusion_matrix()
    if cm_path:
        st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix image not found.")

st.markdown('<div class="section-title">📝 Customer Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Details**")
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)

with col2:
    st.markdown("**Account Details**")
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=100.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=60000.0, step=500.0, format="%.2f")

with col3:
    st.markdown("**Product & Activity**")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
    has_cr_card = st.selectbox("Has Credit Card", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_button = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

if predict_button:
    if model is None or preprocessor is None:
        st.error("Model or preprocessor could not be loaded. Please check the file paths.")
    else:
        input_data = pd.DataFrame({
            "CreditScore": [int(credit_score)],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [int(age)],
            "Tenure": [int(tenure)],
            "Balance": [float(balance)],
            "NumOfProducts": [int(num_of_products)],
            "HasCrCard": [int(has_cr_card)],
            "IsActiveMember": [int(is_active_member)],
            "EstimatedSalary": [float(estimated_salary)]
        })

        try:
            transformed_input = preprocessor.transform(input_data)
            churn_probability = model.predict_proba(transformed_input)[0][1]
            prediction = model.predict(transformed_input)[0]

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(label="Churn Probability", value=f"{churn_probability * 100:.2f}%")

            with res_col2:
                st.metric(label="Retention Probability", value=f"{(1 - churn_probability) * 100:.2f}%")

            with res_col3:
                prediction_label = "🔴 Churn" if prediction == 1 else "🟢 Not Churn"
                st.metric(label="Final Prediction", value=prediction_label)

            if churn_probability > 0.7:
                st.markdown(
                    '<div class="prediction-box churn">⚠️ High churn risk. Recommend immediate retention strategy.</div>',
                    unsafe_allow_html=True
                )
            elif 0.4 <= churn_probability <= 0.7:
                st.markdown(
                    '<div class="prediction-box" style="background-color:#fff8e1;border:2px solid #f39c12;color:#d68910;">⚡ Moderate risk. Monitor customer activity.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-box no-churn">✅ Low churn risk. Customer likely to stay.</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>Bank Customer Churn Predictor | Powered by Machine Learning | For Internal Use Only</div>",
    unsafe_allow_html=True
)