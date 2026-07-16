import pickle
import mlflow
import mlflow.sklearn

from preprocessing import preprocess_text

# Tracking URI.
mlflow.set_tracking_uri(
    "https://dagshub.com/kush2501/mlops-mini-project.mlflow"
)


# -------------------- Load Model -------------------- #
model = mlflow.sklearn.load_model("models:/model/Production")

print("Expected Features:", model.n_features_in_)

# -------------------- Load Vectorizer -------------------- #

with open("../artifacts/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
   
    print(len(vectorizer.vocabulary_))


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