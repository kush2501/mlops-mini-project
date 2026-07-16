import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn

from .preprocessing import preprocess_text

# -------------------- Tracking URI -------------------- #

mlflow.set_tracking_uri(
    "https://dagshub.com/kush2501/mlops-mini-project.mlflow"
)

# -------------------- Project Root -------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------- Load Model -------------------- #

model = mlflow.sklearn.load_model(
    "models:/model/Production"
)

print("Expected Features:", model.n_features_in_)

# -------------------- Load Vectorizer -------------------- #

VECTORIZER_PATH = BASE_DIR / "artifacts" / "vectorizer.pkl"

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

print("Vocabulary Size:", len(vectorizer.vocabulary_))

# -------------------- Predict -------------------- #

def predict_sentiment(text: str):

    # -------------------- Preprocessing -------------------- #

    clean_text = preprocess_text(text)

    print(f"Original Text  : {text}")
    print(f"Processed Text : {clean_text}")

    # -------------------- Vectorization -------------------- #

    text_vector = vectorizer.transform([clean_text])

    # -------------------- Prediction -------------------- #

    prediction = model.predict(text_vector)[0]

    probability = model.predict_proba(text_vector)[0]

    confidence = round(max(probability) * 100, 2)

    sentiment = "😊 Positive" if prediction == 1 else "😔 Negative"

    return {
        "prediction": sentiment,
        "confidence": confidence,
        "processed_text": clean_text
    }