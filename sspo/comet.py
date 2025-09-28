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

    xgb_reg = load_xgb_reg(filename)



def load_xgb_reg(filename: str) -> XGBRegressor:
    """Loads the trained model in the cloud to a local file using pickle."""
    print("\n--- Loading Model ---")

    BUCKET_NAME = os.environ["BUCKET_NAME"]
    storage_filename = f"models/{filename}.pkl"
    local_filename = "model.pkl"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)

    file_path = Path(local_filename)
    xgb_reg = pickle.load(open(file_path, 'rb'))

    print(f"📎 Model {filename}.pkl successfully loaded from Google Cloud into '{file_path}'")

    return xgb_reg


if __name__ == "__main__":
    print("=============================================")
    print("            Starting comet.py           ")
    print("=============================================")

    run_comet_experiment("model_500m_no_power_max")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
