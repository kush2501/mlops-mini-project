from flask import Flask, render_template, request
from predictor import predict_sentiment

app = Flask(__name__)


# -------------------- Home -------------------- #

@app.route("/")
def home():

    return render_template("index.html")


# -------------------- Prediction -------------------- #

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    result = predict_sentiment(text)

    return render_template(
        "index.html",
        prediction=result["prediction"],
        confidence=result["confidence"],
        processed_text=result["processed_text"],
        original_text=text
    )


# -------------------- Run App -------------------- #

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )