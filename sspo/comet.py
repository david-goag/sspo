import os
from comet_ml import start
import xgboost as xgb
from sspo.registry.load_model import load_xgb_reg

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
