# train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle

def load_data(path: str) -> pd.DataFrame:
    """Loads data from a Parquet file and sets the index."""
    file_path = Path(path)
    df = pd.read_parquet(file_path)
    df.set_index("id", inplace=True)

    print(f"✅ Data loaded successfully from '{file_path}'.")
    print(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df

def fit_xgboost(df: pd.DataFrame) -> XGBRegressor:
    """Splits data, trains an XGBoost model, and evaluates it."""
    print("\n--- 2. Preparing Data ---")
    X = df.drop(columns=['time'])
    y = df.time

    # First split: separate 30% for testing/validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Second split: divide the 30% into half for testing and half for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"✅ Data split into training, validation, and test sets.")
    print(f"   - Training set size: {len(X_train)} samples")
    print(f"   - Validation set size: {len(X_val)} samples")
    print(f"   - Test set size: {len(X_test)} samples")

    print("\n--- 3. Training Model ---")
    print("⏳ Fitting XGBoost Regressor...")
    xgb_reg = XGBRegressor(max_depth=20, n_estimators=1000, eval_metric=["rmse"], learning_rate=0.1, early_stopping_rounds=5)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    rmse = round(mean_squared_error(y_val.values, xgb_reg.predict(X_val))**0.5, 2)
    print(f"✅ Model training complete.")
    print(f"📊 Validation RMSE: {rmse} seconds")

    return xgb_reg

def predict_xgboost(xgb_reg: XGBRegressor, X_pred: pd.DataFrame) -> float:
    """Makes a single prediction using the trained model."""
    print("\n--- 4. Making a Sample Prediction ---")
    y_pred = float(xgb_reg.predict(X_pred)[0])

    # Convert seconds to a more readable format (MM:SS)
    minutes = int(y_pred // 60)
    seconds = int(y_pred % 60)

    print(f"✅ Predicted time for the first row: {minutes:02d}:{seconds:02d} minutes")

    return y_pred

def save_xgb_reg(xgb_reg: XGBRegressor, path: str) -> None:
    """Saves the trained model to a file using pickle."""
    print("\n--- 5. Saving Model ---")
    file_path = Path(path)
    pickle.dump(xgb_reg, open(file_path, "wb"))

    print(f"💾 Model successfully saved to '{file_path}'")


if __name__ == "__main__":
    print("=============================================")
    print("   Starting Strava Power Model Training      ")
    print("=============================================")

    print("\n--- 1. Loading Data ---")
    df = load_data("database/current/10475048_data.parquet")

    xgb_reg = fit_xgboost(df)

    predict_xgboost(xgb_reg, df.iloc[[0]].drop(columns=['time']))

    save_xgb_reg(xgb_reg, "xgb_reg.pkl")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
