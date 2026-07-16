import os
import pickle
import logging
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression


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

# -------------------- Project Root -------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# -------------------- Load Data -------------------- #

def load_data():
    """
    Load train feature dataset.
    """

    try:

        logger.info("Loading feature dataset...")

        TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_bow.csv"

        train_data = pd.read_csv(TRAIN_DATA_PATH)

        logger.info(f"Train Shape : {train_data.shape}")

        return train_data

    except Exception:

        logger.exception("Failed to load feature dataset.")
        raise


# -------------------- Split Features & Target -------------------- #

def split_features_target(train_df):
    """
    Split features and target.
    """

    try:

        logger.info("Splitting features and target...")

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        logger.info("Feature-target split completed.")

        return X_train, y_train

    except Exception:

        logger.exception("Error while splitting features and target.")
        raise


# -------------------- Train Model -------------------- #

def train_model(X_train, y_train):
    """
    Train Logistic Regression model.
    """

    try:

        logger.info("Training Logistic Regression Model...")

        model = LogisticRegression(
            C=1,
            solver="liblinear",
            penalty="l2"
        )

        model.fit(X_train, y_train)

        logger.info("Model trained successfully.")

        return model

    except Exception:

        logger.exception("Model training failed.")
        raise


# -------------------- Save Model -------------------- #

def save_model(model):
    """
    Save trained model.
    """

    try:

        ARTIFACTS_DIR = BASE_DIR / "artifacts"
        ARTIFACTS_DIR.mkdir(exist_ok=True)

        MODEL_PATH = ARTIFACTS_DIR / "model.pkl"

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(model, file)
            

        logger.info(f"Model saved successfully at {MODEL_PATH}")


    except Exception:

        logger.exception("Failed to save model.")
        raise


# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Building Pipeline Started")

        train_data = load_data()

        X_train, y_train = split_features_target(
            train_data
        )

        model = train_model(
            X_train,
            y_train
        )

        save_model(model)

        logger.info("Model Building Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()