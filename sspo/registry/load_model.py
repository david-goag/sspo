import os
from pathlib import Path
from xgboost import XGBRegressor
import pickle
from google.cloud import storage


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
    loaded_object = pickle.load(open(file_path, 'rb'))

    # Handle different save formats
    if isinstance(loaded_object, XGBRegressor):
        # Already the correct type
        xgb_reg = loaded_object
        print(f"📎 Model {filename}.pkl loaded as XGBRegressor")
    elif isinstance(loaded_object, dict) and 'model' in loaded_object:
        # Extract model from dictionary
        xgb_reg = loaded_object['model']
        print(f"📎 Model {filename}.pkl extracted from dictionary")
    else:
        raise TypeError(f"Unexpected model format: {type(loaded_object)}")

    print(f"📎 Model {filename}.pkl successfully loaded from Google Cloud into '{file_path}'")

    return xgb_reg


if __name__ == "__main__":
    print("=============================================")
    print("            Starting load_model.py           ")
    print("=============================================")

    load_xgb_reg("xgb_reg_24_54")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
