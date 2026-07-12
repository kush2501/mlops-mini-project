# 📘 Project Development Notes

## Project Name

End-to-End Sentiment Analysis MLOps Pipeline

---

# Development Journey

This project was developed by following a complete MLOps workflow, starting from data preprocessing and ending with a deployed Flask web application.

The objective was not only to train a Machine Learning model but also to build a production-ready pipeline using modern MLOps tools.

---

# Completed Components

✅ Data Ingestion

- Load raw dataset
- Split into train and test data

---

✅ Data Preprocessing

- Lower case conversion
- Stopword removal
- Number removal
- URL removal
- Punctuation removal
- Lemmatization

---

✅ Feature Engineering

- CountVectorizer
- Vocabulary generation
- Saved vectorizer.pkl for inference

---

✅ Model Building

- Logistic Regression
- Model serialization
- model.pkl saved

---

✅ Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score

Evaluation metrics are saved into:

reports/metrics.json

---

✅ MLflow Experiment Tracking

Implemented:

- Parameters Logging
- Metrics Logging
- Model Logging
- Artifact Logging

---

✅ Model Registry

Model registration is performed using:

run_info.json

Every successful experiment creates a new registered model version.

---

✅ Flask Deployment

Developed a complete Flask application including:

- HTML Frontend
- Prediction API
- Professional UI
- Loading Spinner
- Confidence Score
- Prediction Badge

---

# Folder Artifacts

The following artifacts are generated during the pipeline:

artifacts/

- model.pkl
- vectorizer.pkl
- run_info.json

reports/

- metrics.json

---

# Known Issues

## 1. DagsHub Model Registry Stage Promotion

The model is successfully registered into the DagsHub MLflow Model Registry.

However, automatic transition of newly registered versions into **Staging** is currently not available in my setup.

Current workflow:

Register Model

↓

Manual Stage Transition

↓

Staging

↓

Production

Stage promotion is therefore performed manually through the DagsHub UI.

This does not affect model registration or experiment tracking.

---

## 2. Package Compatibility

During development several package compatibility issues were encountered.

Examples:

- mlflow
- botocore
- aiobotocore
- sklearn
- mlxtend

These were resolved by selecting mutually compatible package versions.

---

## 3. Windows Development Environment

The project was developed on Windows.

Docker deployment will provide an operating-system-independent execution environment.

---

# Lessons Learned

This project provided practical experience in:

- Git
- DVC Pipelines
- MLflow Tracking
- Model Registry
- Artifact Management
- Flask Deployment
- Machine Learning Pipeline Design
- Debugging Dependency Conflicts
- End-to-End MLOps Workflow

---

# Future Improvements

The following improvements are planned:

- Automatic Model Registry Promotion
- FastAPI Deployment
- Docker Containerization
- AWS EC2 Deployment
- GitHub Actions CI/CD
- Kubernetes Deployment
- Monitoring
- Model Drift Detection
- Automated Retraining Pipeline

---

# Current Workflow

Data

↓

Data Ingestion

↓

Preprocessing

↓

Feature Engineering

↓

Model Training

↓

Model Evaluation

↓

MLflow Tracking

↓

Model Registry

↓

Flask Web Application

↓

Docker

↓

AWS Deployment

↓

CI/CD

↓

Kubernetes

---

# Project Status

✔ Data Pipeline Completed

✔ ML Pipeline Completed

✔ DVC Pipeline Completed

✔ MLflow Tracking Completed

✔ Model Registry Completed

✔ Flask Web Application Completed

⏳ Docker (In Progress)

⏳ AWS Deployment

⏳ CI/CD

⏳ Kubernetes

---

Last Updated

July 2026