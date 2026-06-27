"""
app_pages/classification.py
------------------------------
Interactive customer churn prediction page.
Loads the trained classification pipeline and lets the user input
customer attributes to get a live churn prediction + probability.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import add_churn_features  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def render():
    st.title("📉 Customer Churn Predictor")
    st.markdown("Predicts whether a customer is likely to **churn** using a trained "
                "classification pipeline.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run `python src/train_classification.py` first.")
        return

    model = load_model()

    st.markdown("### Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 15.0, 150.0, 65.0, step=1.0)
        support_calls = st.slider("Support Calls (last period)", 0, 10, 1)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
        )

    input_df = pd.DataFrame([{
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Contract_Type": contract,
        "Internet_Service": internet,
        "Tech_Support": tech_support,
        "Payment_Method": payment,
        "Support_Calls": support_calls,
    }])

    # Apply the SAME feature engineering used at training time
    input_df = add_churn_features(input_df)

    st.markdown("---")
    if st.button("🔮 Predict Churn", type="primary"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"### ⚠️ Likely to Churn — Probability: **{probability:.1%}**")
        else:
            st.success(f"### ✅ Likely to Stay — Churn Probability: **{probability:.1%}**")

        st.progress(min(max(probability, 0.0), 1.0))

        with st.expander("See engineered feature values sent to the model"):
            st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Pipeline used:**
    `Raw Input → Feature Engineering → ColumnTransformer (Scaling + OneHotEncoding) → Trained Classifier`
    """)
