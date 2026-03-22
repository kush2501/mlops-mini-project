from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle

mlflow.set_tracking_uri("https://dagshub.com/kush2501/mlops-mini-project.mlflow")
dagshub.init(repo_owner='kush2501', repo_name='mlops-mini-project', mlflow=True)


app = Flask(__name__)

# Load model from model registry.
model_name = "model"
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():

    text = request.form['text']

    # clean the data.
    text = normalize_text(text)

    # BOW.
    features = vectorizer.transform([text])

    # Prediction.
    result = model.predict(features)


    # Result.
    return str(result[0])




app.run(debug=True)