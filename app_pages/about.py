"""
app_pages/about.py
---------------------
Explains the full end-to-end ML workflow used in this project,
useful both as in-app documentation and LinkedIn talking points.
"""

import streamlit as st


def render():
    st.title("ℹ️ About OmniPredict AI & Project Workflow")

    st.markdown("""
    **OmniPredict AI** is an end-to-end machine learning project demonstrating the
    complete lifecycle of building and deploying ML models — covering both
    **regression** and **classification** problem types in a single unified app.

    ---
    ### 🔄 End-to-End Workflow

    **1. Data Generation / Ingestion**
    - Two datasets are used: a housing dataset (regression target: `SalePrice`)
      and a telecom-style churn dataset (classification target: `Churn`).
    - Located in `data/`, generated via `src/generate_data.py`.

    **2. Preprocessing**
    - Missing value imputation (`SimpleImputer`)
    - Numeric scaling (`StandardScaler`)
    - Categorical encoding (`OneHotEncoder`)
    - All bundled into a single reusable `ColumnTransformer` (see `src/preprocessing.py`)
      so training and inference always use identical transformations.

    **3. Feature Engineering**
    - House data: `Total_Rooms`, `Is_New`, `Has_Garage`
    - Churn data: `Avg_Charge_per_Month`, `High_Support_Calls`, `Is_New_Customer`

    **4. Model Training & Selection**
    - Regression candidates: `LinearRegression`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`
    - Classification candidates: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
    - Each is trained inside an sklearn `Pipeline` (preprocessor + model) and evaluated
      on a held-out test set. The best model is automatically selected and saved.

    **5. Model Evaluation**
    - Regression: MAE, RMSE, R²
    - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
    - Full comparison tables available on the **Model Evaluation** page.

    **6. Deployment**
    - The winning pipelines (`house_price_model.pkl`, `churn_model.pkl`) are loaded
      directly into this **Streamlit** app.
    - Users interact with live sliders/inputs and get real-time predictions —
      no separate API server required for this version.

    ---
    ### 🧱 Tech Stack
    - **Python**, **pandas**, **NumPy**
    - **scikit-learn** (pipelines, preprocessing, modeling, metrics)
    - **Streamlit** (multi-page interactive UI)
    - **Matplotlib / Seaborn** (data visualization)
    - **joblib** (model persistence)

    ---
    ### 📌 Why this project matters
    This project was built to demonstrate practical, production-style ML engineering:
    - Reproducible pipelines (no train/serve skew)
    - Multiple problem types in one architecture (regression + classification)
    - Clean separation of concerns (`src/` for logic, `app_pages/` for UI)
    - A real, interactive deployment — not just a notebook.
    """)

    st.markdown("---")
    st.caption("👨‍💻 Built as a portfolio project to showcase a complete ML workflow, from raw data to deployment.")
