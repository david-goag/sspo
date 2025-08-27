import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle

def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path) # "database/current/10475048_data.parquet"
    df = pd.read_parquet(file_path)
    df.set_index("id", inplace=True)

    print(f"Data loaded from {file_path}")

    return df

def fit_xgboost(df: pd.DataFrame) -> XGBRegressor:
    X = df.drop(columns=['time'])
    y = df.time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)
    xgb_reg = XGBRegressor(max_depth=20, n_estimators=1000, eval_metric=["rmse"], learning_rate=0.1, early_stopping_rounds=5)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    print(f"XGBoost Regressor fitted with {round(mean_squared_error(y_val.values, xgb_reg.predict(X_val))**0.5, 2)} rmse")

    return xgb_reg

def predict_xgboost(xgb_reg: XGBRegressor, X_pred: pd.DataFrame) -> float:
    y_pred = float(xgb_reg.predict(X_pred)[0])

    print(f"Predicted {y_pred} seconds")

    return y_pred

def save_xgb_reg(xgb_reg: XGBRegressor, path: str) -> None:
    file_path = Path(path) # "xgb_reg.pkl"
    pickle.dump(xgb_reg, open(file_path, "wb"))

    print(f"XGBoost Regressor saved to {file_path}")


if __name__ == "__main__":
    df = load_data("database/current/10475048_data.parquet")
    xgb_reg = fit_xgboost(df)
    predict_xgboost(xgb_reg, df.iloc[[0]].drop(columns=['time']))
    save_xgb_reg(xgb_reg, "xgb_reg.pkl")
