import os
import json
import logging

import mlflow
import dagshub

from mlflow import MlflowClient

# -------------------- Logger -------------------- #

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "project.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -------------------- Load Run Information -------------------- #

def load_run_info():
    """
    Load Run ID and Model Name from artifacts/run_info.json
    """

    try:

        logger.info("Loading Run Information...")

        with open("./artifacts/run_info.json", "r") as file:
            run_info = json.load(file)

        logger.info("Run Information Loaded Successfully.")

        return run_info

    except Exception:

        logger.exception("Failed to load run_info.json")
        raise


# -------------------- Register Model -------------------- #

def register_model(run_info):
    """
    Register MLflow model using Run ID.
    """

    try:

        client = MlflowClient()

        run_id = run_info["run_id"]
        model_name = run_info["model_name"]

        model_uri = f"runs:/{run_id}/{model_name}"

        logger.info(f"Run ID      : {run_id}")
        logger.info(f"Model URI   : {model_uri}")

        registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
            )

        version = registered_model.version

        logger.info("=" * 60)
        logger.info("Model Registered Successfully")
        logger.info(f"Registered Model : {model_name}")
        logger.info(f"Version          : {version}")
        logger.info("=" * 60)

       

    except Exception:

        logger.exception("Model Registration Failed")
        raise


# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Registry Pipeline Started")

        # -------------------- DagsHub + MLflow -------------------- #

        mlflow.set_tracking_uri(
            "https://dagshub.com/kush2501/mlops-mini-project.mlflow"
        )

        dagshub.init(
            repo_owner="kush2501",
            repo_name="mlops-mini-project",
            mlflow=True
        )

        # -------------------- Load Run Info -------------------- #

        run_info = load_run_info()

        # -------------------- Register Model -------------------- #

        register_model(run_info)

        logger.info("Model Registry Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Pipeline Failed")
        raise


if __name__ == "__main__":
    main()