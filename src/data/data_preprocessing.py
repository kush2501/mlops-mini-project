import numpy as np
import pandas as pd

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
import logging

# Downlaod NLTK resources.
nltk.download('wordnet')
nltk.download('stopwords')


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

# Read Data.
def load_data()-> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Loading train and test datasets......")

        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        logger.info(
            f"Train data loaded sucessfully. Shape:{train_data.shape}")
        
        logger.info(
            f"Test data loaded sucessfully. Shape: {test_data.shape}"
        )

        return train_data, test_data
    
    except Exception :
        logger.exception("Error while loading datasets.")
        raise

# Text Cleaning Functions.
def lemmatization(text:str)->str:
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text:str)->str:
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text:str)->str:
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text:str)->str:

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text:str)->str:
    ## Remove punctuations
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text:str)->str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df:pd.DataFrame)->None:
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df:pd.DataFrame)->pd.DataFrame:

    logger.info("Starting text normalization.....")
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    
    logger.info("Text normalization completed.")
    return df

# Remove Empty Rows.
def remove_empty_rows(df:pd.DataFrame)->pd.DataFrame:

    before = len(df)

    df['content'] = df['content'].fillna("")

    df = df[df['content'].str.strip() != ""]

    after = len(df)

    logger.info(f"Removed {before-after}empty rows.")

    return df

# Save Data.
def save_data(train_df:pd.DataFrame, test_df:pd.DataFrame)->None:

    try:

        logger.info("Saving processed datasets....")

        data_path = os.path.join("data", "interim")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.info("Prcessed datasets saved sucessfully.")

    except Exception:
        logger.exception("Error while saving processed datasets")
        raise

# Main Functions.

def main():
    try:

        logger.info("=" * 50)
        logger.info("Text preprocessing pipeline started")

        train_data, test_data = load_data()
        
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        train_processed = remove_empty_rows(train_processed_data)
        test_processed = remove_empty_rows(test_processed_data)

        save_data(train_processed, test_processed)

        logger.info(f"Train Shape : {train_processed.shape}")
        logger.info(f"Test Shape: {test_processed.shape}")
        
        logger.info("Pipeline Completed Sucessfully.")
        logger.info("=" * 50)

    except Exception:
        logger.exception("Pipeline Execution Falied")
        raise

if __name__=="__main__":
    main()         
    



  