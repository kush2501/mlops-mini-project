import mlflow

run_id = "d562c30a868c435598918b3831b5be82"

artifacts = mlflow.artifacts.list_artifacts(run_id)

for artifact in artifacts:
    print(artifact.path)