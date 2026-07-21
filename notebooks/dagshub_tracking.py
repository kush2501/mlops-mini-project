import mlflow
import dagshub
dagshub.init(repo_owner='kush2501', repo_name='emotion-detection-using-mlflow-dvc', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/kush2501/emotion-detection-using-mlflow-dvc.mlflow")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)