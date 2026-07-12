import pickle

from preprocessing import preprocess_text

# -------------------- Load Model -------------------- #

with open("../artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------- Load Vectorizer -------------------- #

with open("../artifacts/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


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