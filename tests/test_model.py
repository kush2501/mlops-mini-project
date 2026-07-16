import os
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri(
    "https://dagshub.com/kush2501/mlops-mini-project.mlflow"
)


def test_model_loading():
    """
    Check whether Production model loads successfully.
    """

    model = mlflow.sklearn.load_model(
        "models:/model/Production"
    )

    assert model is not None