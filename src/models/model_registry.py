import json 
import mlflow
import logging
import os
import dagshub


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Abhi-0088"
repo_name = "dagshub-demo"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


logger = logging.getLogger("model_registry")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_registry_error.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path:str)->dict:
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
        logger.debug("Model Info successfuly load from %s",file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found in the give path %s",file_path)
        raise
    except Exception as e:
        logger.error("Error Occured while loading the model info %s",e)
        raise

def register_model(model_name:str,model_info:dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        #Register the model
        model_version = mlflow.register_model(model_uri,model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging"
        )

        logger.debug(f"Model {model_name} version {model_version.version} registered and transition to Staging")
    except Exception as e:
        logger.error("Error during model registraion: %s",e)
        raise

def main():
    try:
        model_info_path = "model/experiments_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "XGBoost_Model_1"
        register_model(model_name,model_info)
    except Exception as e:
        logger.error("Failed to complete model registration process: %s",e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()