# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        remote_server_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(remote_server_uri)

        mlflow.sklearn.log_model(lr, "model")

        predictions = lr.predict(test_x)
        signature = infer_signature(test_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            registered_model_name = "ElasticnetWineModel"
            try:
                model_version = mlflow.register_model(
                    model_uri=mlflow.current_run().info.artifact_uri + "/model",
                    name=registered_model_name,
                    signature=signature,
                    input_example=test_x.iloc[0:1],
                )
            except mlflow.exceptions.MlflowException:
                model_version = mlflow.register_model(
                    model_uri=mlflow.current_run().info.artifact_uri + "/model",
                    name=registered_model_name,
                )

            # Create a list of dict to define the stage for each model version
            stages = [{"name": "Staging", "current_stage": "None"}, {"name": "Production", "current_stage": "None"}]

            # Transition the model version stage
            for stage in stages:
                try:
                    mlflow.transition_model_version_stage(
                        name=registered_model_name,
                        version=model_version.version,
                        stage=stage["name"],
                    )
                except mlflow.exceptions.MlflowException:
                    continue

            print(f"Registered model name: {registered_model_name} and version: {model_version.version}")
        else:
            print(f"Model logged in run {mlflow.current_run().info.run_id}")
