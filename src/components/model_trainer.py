import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from sklearn.model_selection import train_test_split
from src.utils.utils import save_object

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ModelTrainer:
    def __init__ (self,
                  data_transformation_artifacts: DataTransformationArtifacts,
                  model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        self.evalmetrics = []

    def save_metrics(self, y_test, y_pred, model_name):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2_val = r2_score(y_test,y_pred)

        inf = f"""\'Model': {model_name}\n'MSE': {mse}\n'MAE': {mae}\n'R2 Score': {r2_val}\n{'='*35}\n\n"""
        self.evalmetrics.append(inf)         

        return mse, mae, r2_val 


    def model_training(self):
        try:
            X_train = pd.read_csv(self.data_transformation_artifacts.x_train_transform_file_path)
            X_test = pd.read_csv(self.data_transformation_artifacts.x_test_transform_file_path)
            X_train.drop(labels=['Unnamed: 0'], axis=1,inplace=True)
            X_test.drop(labels=['Unnamed: 0'], axis=1,inplace=True)

            y_train = np.load(self.data_transformation_artifacts.y_train_transform_file_path)
            y_test = np.load(self.data_transformation_artifacts.y_test_transform_file_path)

            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                # 'Lasso': Lasso(),
                # 'randomforest': RandomForestRegressor(random_state=0),
                # 'gradientboost': GradientBoostingRegressor(random_state=0),
                'ElasticNet': ElasticNet()
                # 'XGBRegressor': XGBRegressor()
            }

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            metrics = []
            for model_name in models.keys():
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse, mae, r2_val  = self.save_metrics(y_test, y_pred, model_name)
                values = {
                    'model': model_name,
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2_val
                }

                metrics.append(values)
            min_err=1.01
            best_model_name=None
            for metric in metrics:
                if min_err > metric[BEST_MODEL_CRITERIA]:
                    min_err = metric[BEST_MODEL_CRITERIA]
                    best_model_name = metric['model']

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            
            save_object(file_path=self.model_trainer_config.MODEL_FILE_PATH, object=best_model)
            save_object(file_path=os.path.join(MODEL_FOLDER, MODEL_FILE), object=best_model)

            with open(self.model_trainer_config.METRICS_FILE_PATH, 'w') as file:
                for entry in self.evalmetrics:
                    file.write(entry)
            
            model_trainer_artifact = ModelTrainerArtifacts(
                model_file_path=self.model_trainer_config.MODEL_FILE_PATH,
                metrics_file_path=self.model_trainer_config.METRICS_FILE_PATH
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_training(self):
        try:
            model_trainer_artifact = self.model_training()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
