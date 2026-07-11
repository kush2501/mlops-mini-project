import os
import yaml
import pandas as pd
import logging

from sklearn.feature_extraction.text import CountVectorizer


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

# Load Parameters
def load_params(params_path: str) -> int:

    try:
        logger.info("Loading parameters from params.yaml...")

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        max_features = params["feature_engineering"]["max_features"]

        logger.info(f"max_features = {max_features}")

        return max_features

    except Exception:
        logger.exception("Failed to load parameters.")
        raise


# Load Processed Data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Loading processed datasets...")

        train_data = pd.read_csv("./data/interim/train_processed.csv")
        test_data = pd.read_csv("./data/interim/test_processed.csv")

        logger.info(f"Train Shape : {train_data.shape}")
        logger.info(f"Test Shape : {test_data.shape}")

        return train_data, test_data

    except Exception:
        logger.exception("Failed to load processed datasets.")
        raise


# Apply Bag of Words
def apply_bow(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Applying BagofWords...")

        X_train = train_df["content"].values
        y_train = train_df["sentiment"].values

        X_test = test_df["content"].values
        y_test = test_df["sentiment"].values

        vectorizer = CountVectorizer(
            max_features=max_features
        )

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow= vectorizer.transform(X_test)

        train_bow = pd.DataFrame(X_train_bow.toarray())
        train_bow["label"] = y_train

        test_bow = pd.DataFrame(X_test_bow.toarray())
        test_bow["label"] = y_test

        logger.info("Bow completed.")
        logger.info(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")

        return train_bow, test_bow

    except Exception:
        logger.exception("Error while applying Bag of Words.")
        raise


# Save Feature Data
def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        logger.info("Saving feature-engineered datasets...")

        data_path = os.path.join("data", "processed")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train_bow.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test_bow.csv"),
            index=False
        )

        logger.info("Datasets saved successfully.")

    except Exception:
        logger.exception("Failed to save feature datasets.")
        raise


# Main Function
def main():

    try:

        logger.info("=" * 60)
        logger.info("Feature Engineering Pipeline Started")

        max_features = load_params("params.yaml")

        train_data, test_data = load_data()

        train_bow, test_bow = apply_bow(
            train_data,
            test_data,
            max_features
        )

        save_data(train_bow, test_bow)

        logger.info(
            f"Train Feature Shape : {train_bow.shape}"
        )

        logger.info(
            f"Test Feature Shape : {test_bow.shape}"
        )

        logger.info("Feature Engineering Completed Successfully")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Feature Engineering Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()