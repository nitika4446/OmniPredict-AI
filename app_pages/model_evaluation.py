"""
app_pages/model_evaluation.py
-------------------------------
Displays the model comparison tables produced during training, so the
user can see how each candidate model performed before the best one
was selected for deployment.
"""

import streamlit as st
import pandas as pd
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def render():
    st.title("🧪 Model Evaluation")
    st.markdown("Comparison of all candidate models tried during training, "
                "before the best-performing one was saved for deployment.")

    tab1, tab2 = st.tabs(["Regression Models", "Classification Models"])

    with tab1:
        st.subheader("House Price Prediction — Regression")
        path = os.path.join(MODEL_DIR, "house_price_model_comparison.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.dataframe(df.style.highlight_max(subset=["R2"], color="lightgreen"),
                         use_container_width=True)
            best = df.sort_values("R2", ascending=False).iloc[0]
            st.success(f"🏆 Best model: **{best['model_name']}** (R² = {best['R2']:.4f})")

            st.markdown("""
            **Metrics explained:**
            - **MAE** (Mean Absolute Error) — average absolute prediction error in dollars
            - **RMSE** (Root Mean Squared Error) — penalizes larger errors more heavily
            - **R²** (Coefficient of Determination) — proportion of variance explained (closer to 1 is better)
            """)
        else:
            st.warning("Run `python src/train_regression.py` to generate comparison results.")

    with tab2:
        st.subheader("Customer Churn Prediction — Classification")
        path = os.path.join(MODEL_DIR, "churn_model_comparison.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.dataframe(df.style.highlight_max(subset=["ROC_AUC"], color="lightgreen"),
                         use_container_width=True)
            best = df.sort_values("ROC_AUC", ascending=False).iloc[0]
            st.success(f"🏆 Best model: **{best['model_name']}** (ROC-AUC = {best['ROC_AUC']:.4f})")

            st.markdown("""
            **Metrics explained:**
            - **Accuracy** — overall % of correct predictions
            - **Precision** — of predicted churners, how many actually churned
            - **Recall** — of actual churners, how many were correctly identified
            - **F1-Score** — harmonic mean of precision & recall
            - **ROC-AUC** — model's ability to distinguish churn vs. non-churn across thresholds
            """)
        else:
            st.warning("Run `python src/train_classification.py` to generate comparison results.")
