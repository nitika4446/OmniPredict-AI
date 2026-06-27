"""
preprocessing.py
-----------------
Shared preprocessing & feature engineering utilities used by both the
regression (house price) and classification (churn) pipelines.

Keeping this logic in one place means the EXACT same transformations
used at training time are reused at inference time inside the Streamlit
app -- avoiding train/serve skew.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def add_house_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for the house price dataset."""
    df = df.copy()
    df["Price_per_SqFt_Proxy"] = df["Bedrooms"] + df["Bathrooms"]  # rooms density proxy
    df["Total_Rooms"] = df["Bedrooms"] + df["Bathrooms"]
    df["Is_New"] = (df["Age_Years"] <= 5).astype(int)
    df["Has_Garage"] = (df["Garage_Spaces"] > 0).astype(int)
    return df


def add_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for the customer churn dataset."""
    df = df.copy()
    df["Avg_Charge_per_Month"] = df["Monthly_Charges"] / (df["Tenure_Months"] + 1)
    df["High_Support_Calls"] = (df["Support_Calls"] >= 3).astype(int)
    df["Is_New_Customer"] = (df["Tenure_Months"] <= 6).astype(int)
    return df


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds a reusable sklearn ColumnTransformer that:
      - Imputes + scales numeric features
      - Imputes + one-hot encodes categorical features

    This is fit ONCE on training data and saved alongside the model,
    so the Streamlit app applies identical preprocessing at inference time.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


HOUSE_NUMERIC_FEATURES = [
    "Area_SqFt", "Bedrooms", "Bathrooms", "Age_Years",
    "Garage_Spaces", "Total_Rooms", "Is_New", "Has_Garage"
]
HOUSE_CATEGORICAL_FEATURES = ["Location", "Condition"]
HOUSE_TARGET = "SalePrice"

CHURN_NUMERIC_FEATURES = [
    "Tenure_Months", "Monthly_Charges", "Support_Calls",
    "Avg_Charge_per_Month", "High_Support_Calls", "Is_New_Customer"
]
CHURN_CATEGORICAL_FEATURES = [
    "Contract_Type", "Internet_Service", "Tech_Support", "Payment_Method"
]
CHURN_TARGET = "Churn"
