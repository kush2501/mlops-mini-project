import os
import pickle
import json
import logging
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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

# -------------------- Load Test Data -------------------- #

def load_test_data():
    """
    Load test feature dataset.
    """

    try:

        logger.info("Loading test dataset...")

        test_df = pd.read_csv("./data/processed/test_bow.csv")

        logger.info(f"Test Shape : {test_df.shape}")

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        return X_test, y_test

    except Exception:

        logger.exception("Failed to load test dataset.")
        raise


# -------------------- Load Model -------------------- #

def load_model():
    """
    Load trained model.
    """

    try:

        logger.info("Loading trained model...")

        with open("./artifacts/model.pkl", "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully.")

        return model

    except Exception:

        logger.exception("Failed to load model.")
        raise


# -------------------- Evaluate Model -------------------- #

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    try:

        logger.info("Evaluating model...")

        y_pred = model.predict(X_test)

        # Signature.
        signature = infer_signature(
            pd.DataFrame(X_test), y_pred)

        metrics = {

            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))

        }

        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1 Score  : {metrics['f1_score']:.4f}")

        return metrics, signature

    except Exception:

        logger.exception("Error during model evaluation.")
        raise


# -------------------- Save Metrics -------------------- #

def save_metrics(metrics):
    """
    Save evaluation metrics.
    """

    try:

        logger.info("Saving evaluation metrics...")

        os.makedirs("reports", exist_ok=True)

        with open("./reports/metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved successfully.")

    except Exception:

        logger.exception("Failed to save metrics.")
        raise


# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Evaluation Pipeline Started")

        # -------------------- DagsHub + MLflow -------------------- #

        mlflow.set_tracking_uri(
            "https://dagshub.com/kush2501/mlops-mini-project.mlflow"
        )

        dagshub.init(
            repo_owner="kush2501",
            repo_name="mlops-mini-project",
            mlflow=True
        )

        mlflow.set_experiment("dvc-pipeline")

        with mlflow.start_run(run_name="model_evaluation") as run:

            # -------------------- Load Data -------------------- #

            X_test, y_test = load_test_data()

            # -------------------- Load Model -------------------- #

            model = load_model()

            # -------------------- Evaluate Model -------------------- #

            metrics, signature = evaluate_model(
                model,
                X_test,
                y_test
            )

            # -------------------- Save Metrics -------------------- #

            save_metrics(metrics)

            # -------------------- Log Metrics -------------------- #

            mlflow.log_metrics(metrics)

            # -------------------- Log Parameters -------------------- #

            mlflow.log_params(model.get_params())

            # -------------------- Log Model -------------------- #
            input_example = pd.DataFrame(X_test[:5])

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )
            # -------------------- Log Artifacts -------------------- #

            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("artifacts/model.pkl")

            # -------------------- Save Run Information -------------------- #

            os.makedirs("artifacts", exist_ok=True)

            run_info = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "model_name": "model",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open("artifacts/run_info.json", "w") as file:
                json.dump(run_info, file, indent=4)

            logger.info("Run information saved successfully.")

            logger.info("MLflow logging completed successfully.")

        logger.info("Model Evaluation Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()