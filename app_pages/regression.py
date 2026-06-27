"""
app_pages/regression.py
--------------------------
Interactive house price prediction page.
Loads the trained pipeline (preprocessor + model) and lets the user
input feature values to get a live SalePrice prediction.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import add_house_features  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def render():
    st.title("🏘️ House Price Predictor")
    st.markdown("Predicts **SalePrice** using a trained regression pipeline "
                "(preprocessing + feature engineering + model baked in).")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run `python src/train_regression.py` first.")
        return

    model = load_model()

    st.markdown("### Enter property details")
    col1, col2, col3 = st.columns(3)

    with col1:
        area_sqft = st.slider("Area (sq ft)", 400, 6000, 1800, step=50)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)

    with col2:
        age_years = st.slider("Age (years)", 0, 80, 10)
        garage = st.selectbox("Garage Spaces", [0, 1, 2, 3], index=1)
        location = st.selectbox("Location", ["Downtown", "Suburb", "Rural", "Waterfront"])

    with col3:
        condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"], index=2)

    input_df = pd.DataFrame([{
        "Area_SqFt": area_sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age_years,
        "Garage_Spaces": garage,
        "Location": location,
        "Condition": condition,
    }])

    # Apply the SAME feature engineering used at training time
    input_df = add_house_features(input_df)

    st.markdown("---")
    if st.button("🔮 Predict Sale Price", type="primary"):
        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Sale Price: **${prediction:,.0f}**")

        with st.expander("See engineered feature values sent to the model"):
            st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Pipeline used:**
    `Raw Input → Feature Engineering → ColumnTransformer (Scaling + OneHotEncoding) → Trained Regressor`
    """)
