import os
import yaml
import pickle
import logging
import pandas as pd

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

        train_data = pd.read_csv("./data/interim/train_processed.csv")
        test_data = pd.read_csv("./data/interim/test_processed.csv")

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

        logger.info("Feature datasets saved successfully.")

    except Exception:

        logger.exception("Failed to save feature datasets.")
        raise


# -------------------- Save Vectorizer -------------------- #

def save_vectorizer(vectorizer):

    try:

        logger.info("Saving CountVectorizer...")

        os.makedirs("artifacts", exist_ok=True)

        with open("./artifacts/vectorizer.pkl", "wb") as file:
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

        max_features = load_params("params.yaml")

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