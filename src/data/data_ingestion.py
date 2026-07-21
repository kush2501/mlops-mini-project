# Import Libraries.
import pandas as pd

from sklearn.model_selection import train_test_split
import os
import yaml
import logging

import logging

# Create logs folder.
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path.
LOG_FILE = os.path.join(LOG_DIR, "project.log")


# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE), # Save Logs in file.
        logging.StreamHandler() # Show logs in terminal
    ]
)

logger = logging.getLogger(__name__)

def load_params(params_path: str)-> float:
    try:
        logger.info("Loading parameters from params.yaml")

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        test_size = params["data_ingestion"]["test_size"]
        logger.info(f"Test Size : {test_size}")

        return test_size
    
    except Exception as e:
        logger.error(f"Error loading params file: {e}")
        raise
    
def load_dataset(url :str) ->pd.DataFrame:
    try:

        logger.info("Loading dataset.....")

        df = pd.read_csv(url)

        logger.info(f"Dataset Loaded Sucessfully, shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
def process_data(df: pd.DataFrame)-> pd.DataFrame:

    logger.info("Data Preprocessing Started")

    df.drop(columns=['tweet_id'], inplace=True)
    final_df  = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
    final_df['sentiment'] = final_df["sentiment"].replace({"happiness":1, "sadness":0})
    
    logger.info(f"Processed Dataset Shape : {final_df.shape}")
    logger.info("Data Preprocessing Completed")

    return final_df
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame)->None:
    try:

        logger.info("Saving Train and Test Data")

        data_path = os.path.join("data", "raw")

        os.makedirs(data_path)

        train_data.to_csv(
            os.path.join(data_path,"train.csv"), index=False)
        test_data.to_csv(
            os.path.join(data_path,"test.csv"), index=False)
        
        logger.info("Data Saved Sucessfully")
 
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main()-> None:

    try:

        logger.info("=============== Data Ingestion Started ===============")

        test_size = load_params("params.yaml")
        data_frame = load_dataset("https://raw.githubusercontent.com/Giohanny/Twitter-Sentiment-Analysis/refs/heads/master/text_emotion.csv")
        final_df = process_data(data_frame)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        save_data(train_data, test_data)

        logger.info("============= Data Ingestion Completed ============")

    except Exception as e:
        logger.exception("Pipeline Falied")


if __name__ == "__main__":

    main()