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
    Register MLflow model using Run ID
    and move the newly registered model to Staging.
    """

    try:

        client = MlflowClient()

        run_id = run_info["run_id"]
        model_name = run_info["model_name"]

        model_uri = f"runs:/{run_id}/{model_name}"

        logger.info(f"Run ID      : {run_id}")
        logger.info(f"Model URI   : {model_uri}")

        # Register Model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        version = registered_model.version

        logger.info("=" * 60)
        logger.info("Model Registered Successfully")
        logger.info(f"Registered Model : {model_name}")
        logger.info(f"Version          : {version}")

        # -------------------------------------------------------
        # Move Newly Registered Model to Staging
        # -------------------------------------------------------

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )

        logger.info(f"Stage            : Staging")

        # Compare with Production and Promote if Better
        # -------------------------------------------------------
        promote_best_model(model_name, version)
        
        logger.info("=" * 60)

    except Exception:
        logger.exception("Model Registration Failed")
        raise


def promote_best_model(model_name, new_version):
    """
    Compare the newly registered model with the current Production model.
    If the new model has better accuracy, move it to Production and archive
    the old Production model. Otherwise keep it in Staging.
    """

    try:

        client = MlflowClient()

        logger.info("=" * 60)
        logger.info("Checking Production Model")

        # ------------------------------
        # New model accuracy
        # ------------------------------

        new_version_info = client.get_model_version(
            name=model_name,
            version=new_version
        )

        new_run = client.get_run(new_version_info.run_id)

        new_accuracy = float(
            new_run.data.metrics.get("accuracy", 0)
        )

        logger.info(f"New Accuracy : {new_accuracy}")

        # ------------------------------
        # Production model
        # ------------------------------

        production_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        # -----------------------------------
        # First Production Model
        # -----------------------------------

        if len(production_versions) == 0:

            client.transition_model_version_stage(
                name=model_name,
                version=new_version,
                stage="Production"
            )

            logger.info("No Production Model Found.")
            logger.info("Current Model Promoted to Production.")

            return

        # -----------------------------------
        # Existing Production
        # -----------------------------------

        production = production_versions[0]

        production_run = client.get_run(production.run_id)

        production_accuracy = float(
            production_run.data.metrics.get("accuracy", 0)
        )

        logger.info(
            f"Production Accuracy : {production_accuracy}"
        )

        # -----------------------------------
        # Compare Accuracy
        # -----------------------------------

        if new_accuracy > production_accuracy:

            client.transition_model_version_stage(
                name=model_name,
                version=production.version,
                stage="Archived"
            )

            client.transition_model_version_stage(
                name=model_name,
                version=new_version,
                stage="Production"
            )

            logger.info("Better Model Found")
            logger.info("Old Production Archived")
            logger.info("New Model Promoted to Production")

        else:

            logger.info("Current Production Model is Better")
            logger.info("New Model remains in Staging")

    except Exception:

        logger.exception("Promotion Failed")
        raise

# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Registry Pipeline Started")


        # -------------------- DagsHub Authentication -------------------- #

        os.environ["MLFLOW_TRACKING_USERNAME"] = "kush2501"

        token = os.getenv("DAGSHUB_TOKEN")

        if token is None:
            raise ValueError("DAGSHUB_TOKEN environment variable is not set.")

        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Register token with DagsHub client
        dagshub.auth.add_app_token(token)

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