import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from src.components.data_transformation import DataTransformationArtifacts
from src.components.model_trainer import ModelTrainerArtifacts

class ModelEvaluation:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_artifacts: ModelTrainerArtifacts):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        # self.rmse = None
        # self.mae = None
        # self.r2 = None

    def eval_metrics(self, actual, pred):
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            logging.info("evaluation metrics captured")
            return rmse, mae, r2
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_model_evaluation(self):
        try:
            model_path = self.model_trainer_artifacts.model_file_path
            model = load_object(model_path)
            X_test = pd.read_csv(self.data_transformation_artifacts.x_test_transform_file_path)
            X_test.drop(labels=['Unnamed: 0'], axis=1,inplace=True)
            y_test = np.load(self.data_transformation_artifacts.y_test_transform_file_path)

            prediction = model.predict(X_test)
            rmse,mae,r2 = self.eval_metrics(y_test,prediction)

            params = model.get_params()

            mlflow.set_experiment("Model Evaluation")

            # tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():

                # Log the hyperparameters
                mlflow.log_params(params)

                # log the metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Evaluation Info", "Basic LR model for iris data")

                # Infer the model signature
                signature = infer_signature(X_test, model.predict(X_test))

                # Log the model
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="sklearn_model",
                    signature=signature,
                    input_example=X_test,
                    registered_model_name="best_model",
                )
                #  # Model registry does not work with file store
                # if tracking_url_type_store != "file":

                #     # Register the model
                #     # There are other ways to use the Model Registry, which depends on the use case,
                #     # please refer to the doc for more information:
                #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                #     mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # else:
                #     mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise CustomException(e, sys)