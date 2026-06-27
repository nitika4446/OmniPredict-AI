"""
app_pages/home.py
-------------------
Landing page for OmniPredict AI.
"""

import streamlit as st


def render():
    st.title("🤖 OmniPredict AI")
    st.subheader("A unified Machine Learning platform for Regression & Classification")

    st.markdown("""
    Welcome to **OmniPredict AI** — a project built to demonstrate a complete,
    real-world machine learning workflow: from raw data to a deployed,
    interactive web application.

    ### What this project demonstrates
    | Stage | What's covered |
    |---|---|
    | **Data Preprocessing** | Missing value imputation, scaling, encoding |
    | **Feature Engineering** | Derived features for both tasks |
    | **Regression** | Predicting continuous house sale prices |
    | **Classification** | Predicting binary customer churn |
    | **Model Evaluation** | MAE, RMSE, R², Accuracy, Precision, Recall, F1, ROC-AUC |
    | **Deployment** | Interactive Streamlit web app with sidebar navigation |

    ### How to use this app
    Use the **sidebar on the left** to navigate between:
    - 📊 **Data Explorer** — inspect the underlying datasets
    - 🏘️ **House Price Predictor** — try the regression model live
    - 📉 **Customer Churn Predictor** — try the classification model live
    - 🧪 **Model Evaluation** — compare model performance metrics
    - ℹ️ **About / Workflow** — full project pipeline explanation
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Tasks", "2", "Regression + Classification")
    with col2:
        st.metric("Models Compared", "7", "Across both tasks")
    with col3:
        st.metric("Deployment", "Streamlit", "Cloud-ready")

    st.info("👈 Start by selecting a page from the sidebar.")
