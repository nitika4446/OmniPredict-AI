"""
generate_data.py
-----------------
Generates two synthetic datasets used by OmniPredict AI:

1. data/house_prices.csv   -> Regression task   (predict SalePrice)
2. data/customer_churn.csv -> Classification task (predict Churn: Yes/No)

Run once before training:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_house_prices(n_samples: int = 2000) -> pd.DataFrame:
    """Synthetic housing dataset with numeric + categorical features."""
    area_sqft = np.random.normal(1800, 650, n_samples).clip(400, 6000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples,
                                 p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4], n_samples,
                                  p=[0.25, 0.40, 0.25, 0.10])
    age_years = np.random.exponential(15, n_samples).clip(0, 80)
    garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.40, 0.35, 0.10])
    location = np.random.choice(
        ["Downtown", "Suburb", "Rural", "Waterfront"], n_samples,
        p=[0.30, 0.40, 0.20, 0.10]
    )
    condition = np.random.choice(
        ["Poor", "Fair", "Good", "Excellent"], n_samples,
        p=[0.05, 0.25, 0.50, 0.20]
    )

    # Base price formula with realistic relationships + noise
    location_premium = location_map = {
        "Downtown": 60000, "Suburb": 20000, "Rural": -20000, "Waterfront": 120000
    }
    condition_premium = {
        "Poor": -30000, "Fair": -10000, "Good": 10000, "Excellent": 40000
    }

    price = (
        area_sqft * 120
        + bedrooms * 8000
        + bathrooms * 12000
        + garage * 7000
        - age_years * 900
        + np.array([location_map[l] for l in location])
        + np.array([condition_premium[c] for c in condition])
        + np.random.normal(0, 25000, n_samples)
    )
    price = price.clip(40000, None).round(-2)  # round to nearest 100

    df = pd.DataFrame({
        "Area_SqFt": area_sqft.round(1),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age_years.round(1),
        "Garage_Spaces": garage,
        "Location": location,
        "Condition": condition,
        "SalePrice": price
    })
    return df


def generate_customer_churn(n_samples: int = 2500) -> pd.DataFrame:
    """Synthetic telecom-style churn dataset with numeric + categorical features."""
    tenure_months = np.random.exponential(24, n_samples).clip(0, 72).round(0)
    monthly_charges = np.random.normal(65, 30, n_samples).clip(15, 150).round(2)
    contract = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"], n_samples,
        p=[0.55, 0.25, 0.20]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber Optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    tech_support = np.random.choice(["Yes", "No"], n_samples, p=[0.4, 0.6])
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n_samples, p=[0.35, 0.20, 0.20, 0.25]
    )
    num_support_calls = np.random.poisson(1.5, n_samples).clip(0, 10)

    # Latent churn probability driven by realistic factors
    contract_risk = {"Month-to-Month": 0.45, "One Year": 0.15, "Two Year": 0.05}
    internet_risk = {"Fiber Optic": 0.25, "DSL": 0.10, "No": 0.02}
    support_risk = {"Yes": -0.10, "No": 0.10}

    churn_score = (
        np.array([contract_risk[c] for c in contract])
        + np.array([internet_risk[i] for i in internet_service])
        + np.array([support_risk[t] for t in tech_support])
        + (monthly_charges - 65) / 300
        - (tenure_months / 200)
        + (num_support_calls * 0.04)
        + np.random.normal(0, 0.08, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-8 * (churn_score - 0.25)))  # sigmoid squashing
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "Tenure_Months": tenure_months,
        "Monthly_Charges": monthly_charges,
        "Contract_Type": contract,
        "Internet_Service": internet_service,
        "Tech_Support": tech_support,
        "Payment_Method": payment_method,
        "Support_Calls": num_support_calls,
        "Churn": churn_label
    })
    return df


if __name__ == "__main__":
    house_df = generate_house_prices()
    churn_df = generate_customer_churn()

    house_path = os.path.join(OUTPUT_DIR, "house_prices.csv")
    churn_path = os.path.join(OUTPUT_DIR, "customer_churn.csv")

    house_df.to_csv(house_path, index=False)
    churn_df.to_csv(churn_path, index=False)

    print(f"✅ House price dataset saved: {house_path}  ({house_df.shape[0]} rows)")
    print(f"✅ Customer churn dataset saved: {churn_path}  ({churn_df.shape[0]} rows)")
