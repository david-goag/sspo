import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comet_ml import start
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

    # ===== FEATURE IMPORTANCE ANALYSIS =====

    # Get feature names
    feature_names = X.columns.tolist()

    # 1. Log Weight importance (how often feature is used)
    weight_importance = xgb_reg.feature_importances_
    weight_dict = dict(zip(feature_names, weight_importance))
    experiment.log_parameters(weight_dict, prefix="importance_weight_")

    # 2. Log Gain importance (average gain when feature is used)
    try:
        booster = xgb_reg.get_booster()
        gain_importance = booster.get_score(importance_type='gain')

        # Convert to feature names
        gain_dict = {}
        for i, feature in enumerate(feature_names):
            gain_dict[feature] = gain_importance.get(f'f{i}', 0)

        experiment.log_parameters(gain_dict, prefix="importance_gain_")

        # Create gain importance plot
        gain_df = pd.DataFrame({
            'feature': feature_names,
            'gain': [gain_dict[f] for f in feature_names]
        }).sort_values('gain', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(gain_df['feature'], gain_df['gain'])
        ax.set_xlabel('Gain (Average improvement in loss)')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance by Gain')
        ax.invert_yaxis()
        plt.tight_layout()
        experiment.log_figure("feature_importance_gain", fig)
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not calculate gain importance: {e}")

    # 3. Calculate SHAP values (shows actual impact on predictions)
    try:
        import shap

        # Use a sample of data for SHAP (it can be slow on large datasets)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)

        # Create SHAP explainer
        explainer = shap.Explainer(xgb_reg, X_train)
        shap_values = explainer(X_sample)

        # Calculate mean absolute SHAP value for each feature
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_dict = dict(zip(feature_names, mean_shap))
        experiment.log_parameters(shap_dict, prefix="importance_shap_")

        # Create SHAP importance plot
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(shap_df['feature'], shap_df['mean_shap'])
        ax.set_xlabel('Mean |SHAP value| (Impact on prediction)')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance by SHAP')
        ax.invert_yaxis()
        plt.tight_layout()
        experiment.log_figure("feature_importance_shap", fig)
        plt.close()

        # Create SHAP summary plot (shows distribution of impacts)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        experiment.log_figure("shap_summary_plot", plt.gcf())
        plt.close()

        print("✅ SHAP analysis completed")

    except ImportError:
        print("⚠️ SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"⚠️ Could not calculate SHAP values: {e}")

    # Print summary
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*60)
    if 'gain_df' in locals():
        print("\nTop features by Gain:")
        print(gain_df.to_string(index=False))
    if 'shap_df' in locals():
        print("\nTop features by SHAP (impact on predictions):")
        print(shap_df.to_string(index=False))

    # Log the model to Comet
    experiment.log_model(filename, "model.pkl")

    print(f"✅ Model {filename} logged to Comet ML successfully")
    experiment.end()


if __name__ == "__main__":
    print("=============================================")
    print("            Starting comet.py           ")
    print("=============================================")

    run_comet_experiment("xgb_reg_5_94")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
