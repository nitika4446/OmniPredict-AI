"""
app_pages/data_explorer.py
----------------------------
Lets the user explore both datasets: shape, summary stats, distributions.
"""

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))


def render():
    st.title("📊 Data Explorer")
    st.markdown("Explore the raw datasets used to train the regression and classification models.")

    dataset_choice = st.selectbox(
        "Choose a dataset to explore",
        ["House Prices (Regression)", "Customer Churn (Classification)"]
    )

    if dataset_choice == "House Prices (Regression)":
        df = load_data("house_prices.csv")
        target = "SalePrice"
    else:
        df = load_data("customer_churn.csv")
        target = "Churn"

    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    tab1, tab2, tab3 = st.tabs(["Preview", "Summary Statistics", "Visualizations"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Column data types**")
        st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)

    with tab2:
        st.markdown("**Numeric feature summary**")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("**Missing values**")
        missing = df.isnull().sum()
        st.dataframe(missing[missing >= 0].rename("missing_count").to_frame(), use_container_width=True)

    with tab3:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Distribution of `{target}`**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            if target in numeric_cols:
                sns.histplot(df[target], kde=True, ax=ax, color="#4C72B0")
            else:
                sns.countplot(x=target, data=df, ax=ax, color="#4C72B0")
            st.pyplot(fig)

        with col2:
            feature = st.selectbox("Pick a feature to visualize", [c for c in df.columns if c != target])
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            if feature in numeric_cols:
                sns.histplot(df[feature], kde=True, ax=ax2, color="#55A868")
            else:
                sns.countplot(x=feature, data=df, ax=ax2, color="#55A868")
                plt.xticks(rotation=30)
            st.pyplot(fig2)

        if len(numeric_cols) > 1:
            st.markdown("**Correlation heatmap (numeric features)**")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)
