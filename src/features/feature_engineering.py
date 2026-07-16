import os
import yaml
import pickle
import logging
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

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

# -------------------- Load Parameters -------------------- #

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


# -------------------- Load Data -------------------- #

def load_data():

    try:

        logger.info("Loading processed datasets...")

        TRAIN_DATA_PATH = BASE_DIR / "data" / "interim" / "train_processed.csv"
        TEST_DATA_PATH = BASE_DIR / "data" / "interim" / "test_processed.csv"

        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)

        logger.info(f"Train Shape : {train_data.shape}")
        logger.info(f"Test Shape : {test_data.shape}")

        return train_data, test_data

    except Exception:

        logger.exception("Failed to load processed datasets.")
        raise


# -------------------- Apply Bag Of Words -------------------- #

def apply_bow(
    train_df,
    test_df,
    max_features
):

    try:

        logger.info("Applying Bag of Words...")

        X_train = train_df["content"].values
        y_train = train_df["sentiment"].values

        X_test = test_df["content"].values
        y_test = test_df["sentiment"].values

        vectorizer = CountVectorizer(
            max_features=max_features
        )

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_bow = pd.DataFrame(X_train_bow.toarray())
        train_bow["label"] = y_train

        test_bow = pd.DataFrame(X_test_bow.toarray())
        test_bow["label"] = y_test

        logger.info("Bag of Words completed.")
        logger.info(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")

        return train_bow, test_bow, vectorizer

    except Exception:

        logger.exception("Error while applying Bag of Words.")
        raise


# -------------------- Save Feature Data -------------------- #

def save_data(train_df, test_df):

    try:

        logger.info("Saving feature datasets...")

        DATA_PATH = BASE_DIR / "data" / "processed"
        DATA_PATH.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(
            DATA_PATH / "train_bow.csv",
            index=False
        )

        test_df.to_csv(
            DATA_PATH / "test_bow.csv",
            index=False
        )

        logger.info("Feature datasets saved successfully.")

    except Exception:

        logger.exception("Failed to save feature datasets.")
        raise


# -------------------- Save Vectorizer -------------------- #

def save_vectorizer(vectorizer):

    try:

        logger.info("Saving CountVectorizer...")

        ARTIFACTS_DIR = BASE_DIR / "artifacts"
        ARTIFACTS_DIR.mkdir(exist_ok=True)

        VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.pkl"

        with open(VECTORIZER_PATH, "wb") as file:
            pickle.dump(vectorizer, file)

        logger.info("Vectorizer saved successfully.")

    except Exception:

        logger.exception("Failed to save vectorizer.")
        raise


# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Feature Engineering Pipeline Started")

        PARAMS_PATH = BASE_DIR / "params.yaml"
        max_features = load_params(PARAMS_PATH)

        train_data, test_data = load_data()

        train_bow, test_bow, vectorizer = apply_bow(
            train_data,
            test_data,
            max_features
        )

        save_data(
            train_bow,
            test_bow
        )

        save_vectorizer(
            vectorizer
        )

        logger.info(f"Train Feature Shape : {train_bow.shape}")
        logger.info(f"Test Feature Shape : {test_bow.shape}")

        logger.info("Feature Engineering Completed Successfully.")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Feature Engineering Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()