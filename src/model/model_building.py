import os
import pickle
import yaml
import logging
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression


import logging
import os

# Create logs folder
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "project.log")

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),   # Save logs in file
        logging.StreamHandler()          # Show logs in terminal
    ]
)

logger = logging.getLogger(__name__)

# -------------------- Load Data -------------------- #
def load_data() -> tuple[pd.DataFrame]:
    """
    Load train and test feature datasets.
    """

    try:
        logger.info("Loading feature datasets...")

        train_data = pd.read_csv("./data/processed/train_bow.csv")
        

        logger.info(f"Train Shape : {train_data.shape}")
       

        return train_data

    except Exception:
        logger.exception("Failed to load feature datasets.")
        raise


# -------------------- Split Features -------------------- #
def split_features_target(
    train_df: pd.DataFrame
) -> tuple:
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
def train_model(
    X_train,
    y_train,
):
    """
    Train Logistic Regression.
    """

    try:
        logger.info("Training Logistic Regression model...")

        clf = LogisticRegression(
            C=1, 
            solver='liblinear',
            penalty='l2'
        )

        clf.fit(X_train, y_train)

        logger.info("Model training completed successfully.")

        return clf

    except Exception:
        logger.exception("Error while training the model.")
        raise


# -------------------- Save Model -------------------- #
def save_model(model) -> None:
    """
    Save trained model.
    """

    try:
        logger.info("Saving trained model...")

        os.makedirs("models", exist_ok=True)

        with open("./models/model.pkl", "wb") as file:
            pickle.dump(model, file)

        logger.info("Model saved successfully.")

    except Exception:
        logger.exception("Error while saving model.")
        raise


# -------------------- Main -------------------- #
def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Training Pipeline Started")


        train_data = load_data()

        X_train,  y_train = split_features_target(
        train_data
        )

        model = train_model(
            X_train,
            y_train
            )
        
        save_model(model)


    except Exception:
        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()