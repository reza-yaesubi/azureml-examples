import os
import argparse
import tempfile
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    credit_df = pd.read_csv(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    # Extracting the label column
    y_train = train_df.pop("default payment next month")

    # convert the dataframe values to array
    X_train = train_df.values

    # Extracting the label column
    y_test = test_df.pop("default payment next month")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    ###################
    #</train the model>
    ###################

    ##########################
    #<save and register model>
    ##########################
    # mlflow.sklearn.log_model() always calls the new MLflow 3.x "logged models" API
    # (/api/2.0/mlflow/logged-models), which Azure ML's tracking server does not implement yet
    # and returns 404 for. Work around this by saving the model locally, then uploading it as a
    # plain run artifact, and registering it from that artifact path.
    print("Logging and registering the model via MLFlow")
    artifact_path = args.registered_model_name
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_path = os.path.join(tmp_dir, artifact_path)
        # Some AzureML curated environments do not include the `skops` package that
        # MLflow now defaults to, so pin the serialization format to pickle.
        # No-code online deployment builds the inference env from this model's conda.yaml
        # and its auto-generated scoring script imports azureml.ai.monitoring.Collector, so
        # azureml-ai-monitoring must be baked into the model's pip requirements or the
        # scoring container crashes on startup (502 liveness probe failure).
        mlflow.sklearn.save_model(
            sk_model=clf,
            path=local_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            extra_pip_requirements=["azureml-ai-monitoring"],
        )
        mlflow.log_artifacts(local_model_path, artifact_path=artifact_path)

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    # mlflow.register_model() looks up "logged models" via search_logged_models(),
    # which also 404s on Azure ML. Use the low-level MlflowClient API instead, which
    # registers directly from the run's artifact location without that lookup.
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(args.registered_model_name)
    except MlflowException:
        pass  # registered model already exists
    client.create_model_version(
        name=args.registered_model_name,
        source=RunsArtifactRepository.get_underlying_uri(model_uri),
        run_id=run_id,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        extra_pip_requirements=["azureml-ai-monitoring"],
    )
    ###########################
    #</save and register model>
    ###########################

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
