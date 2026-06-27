"""
app.py
------
OmniPredict AI — Main Streamlit Application Entry Point.

This file sets up page config and a sidebar navigation menu, then
routes to the appropriate page module inside app_pages/.

Run locally:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="OmniPredict AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🤖 OmniPredict AI")
st.sidebar.markdown("End-to-end ML platform: regression, classification & deployment.")
st.sidebar.markdown("---")

PAGES = {
    "🏠 Home": "home",
    "📊 Data Explorer": "data_explorer",
    "🏘️ House Price Predictor (Regression)": "regression",
    "📉 Customer Churn Predictor (Classification)": "classification",
    "🧪 Model Evaluation": "model_evaluation",
    "ℹ️ About / Workflow": "about",
}


selection = st.sidebar.radio("Use the sidebar to navigate through the application.", list(PAGES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn + Streamlit")
st.sidebar.caption("📁 github.com/yourusername/OmniPredict-AI")

page_key = PAGES[selection]

# ---------------------------------------------------------
# Page Router
# ---------------------------------------------------------
if page_key == "home":
    from app_pages import home
    home.render()

elif page_key == "data_explorer":
    from app_pages import data_explorer
    data_explorer.render()

elif page_key == "regression":
    from app_pages import regression
    regression.render()

elif page_key == "classification":
    from app_pages import classification
    classification.render()

elif page_key == "model_evaluation":
    from app_pages import model_evaluation
    model_evaluation.render()

elif page_key == "about":
    from app_pages import about
    about.render()
