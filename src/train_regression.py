"""
train_regression.py
--------------------
Trains and evaluates multiple regression models on the house price
dataset, selects the best performer, and saves it (along with the
fitted preprocessor) to the models/ directory for use by the Streamlit app.

Run:
    python src/train_regression.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    add_house_features,
    build_preprocessor,
    HOUSE_NUMERIC_FEATURES,
    HOUSE_CATEGORICAL_FEATURES,
    HOUSE_TARGET,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n📊 {name}")
    print(f"   MAE  : {mae:,.2f}")
    print(f"   RMSE : {rmse:,.2f}")
    print(f"   R²   : {r2:.4f}")
    return {"model_name": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = add_house_features(df)

    X = df[HOUSE_NUMERIC_FEATURES + HOUSE_CATEGORICAL_FEATURES]
    y = df[HOUSE_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(HOUSE_NUMERIC_FEATURES, HOUSE_CATEGORICAL_FEATURES)

    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_SEED
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test, name)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n🏆 Model comparison (sorted by R²):")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model_name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n✅ Best model: {best_name}")

    model_path = os.path.join(MODEL_DIR, "house_price_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"💾 Saved best pipeline to {model_path}")

    results_df.to_csv(os.path.join(MODEL_DIR, "house_price_model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
