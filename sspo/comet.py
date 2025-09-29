import os
import pandas as pd
import numpy as np
from comet_ml import start
import xgboost as xgb
from pathlib import Path
from sspo.registry.load_model import load_xgb_reg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_comet_experiment(filename: str) -> None:

    COMET_KEY = os.environ["COMET_KEY"]

    experiment = start(
    api_key=COMET_KEY,
    project_name="sspo_test",
    workspace="david-goag"
    )

    # Load trained model
    xgb_reg = load_xgb_reg(filename)

    # Log model parameters (XGBoost hyperparameters)
    model_params = xgb_reg.get_params()
    experiment.log_parameters(model_params)

    # Log additional metadata
    experiment.log_parameters({
        "model_filename": filename,
        "model_type": "XGBRegressor",
        "framework": "XGBoost"
    })

    # Load train/test data
    file_path = Path("database/test_fit/all_athletes_20250903175112.parquet")
    df = pd.read_parquet(file_path)
    print(f"✅ Data loaded successfully from '{file_path}'. Shape: {df.shape}")
    df = df.drop(columns=["power_max"])
    X = df.drop(columns=["time"])
    y = df.time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Make predictions
    y_pred = xgb_reg.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    experiment.log_metrics({
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2
    })

    # Log the model to Comet
    experiment.log_model(filename, "model.pkl")

    print(f"✅ Model {filename} logged to Comet ML successfully")
    experiment.end()


if __name__ == "__main__":
    print("=============================================")
    print("            Starting comet.py           ")
    print("=============================================")

    run_comet_experiment("xgb_reg_5_84")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
