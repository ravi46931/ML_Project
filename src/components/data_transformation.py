import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


class DataTransformation:
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifacts,
                 data_transformation_config: DataTransformationConfig
                 ):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config
        
    def get_preprocessor_object(self):
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self):
        try:
            df=pd.read_csv(self.data_ingestion_artifacts.data_file_path)

            df.drop(labels=['Unnamed: 0'], axis=1,inplace=True)
            
            X = df.drop([TARGET_COLUMN, 'id'], axis=1)
            y = df[TARGET_COLUMN]
            # Assuming X is your feature matrix and y is your target vector
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


            preprocessor = self.get_preprocessor_object()

            x_train_transform = preprocessor.fit_transform(X_train)
            x_test_transform = preprocessor.transform(X_test)

            X_train=pd.DataFrame(x_train_transform, columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(x_test_transform, columns=preprocessor.get_feature_names_out())

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            X_train.to_csv(self.data_transformation_config.X_TRAIN_TRANSFORM_FILE_PATH)
            X_test.to_csv(self.data_transformation_config.X_TEST_TRANSFORM_FILE_PATH)

            np.save(self.data_transformation_config.Y_TRAIN_TRANSFORM_FILE_PATH, y_train)
            np.save(self.data_transformation_config.Y_TEST_TRANSFORM_FILE_PATH, y_test)

            dump(preprocessor, self.data_transformation_config.PREPROCESSOR_FILE_PATH)

            os.makedirs(MODEL_FOLDER, exist_ok=True)
            dump(preprocessor, os.path.join(MODEL_FOLDER, PREPROCESSOR_FILE))
            
            data_transformation_artifacts = DataTransformationArtifacts(
                x_train_transform_file_path=self.data_transformation_config.X_TRAIN_TRANSFORM_FILE_PATH,
                x_test_transform_file_path=self.data_transformation_config.X_TEST_TRANSFORM_FILE_PATH,
                y_train_transform_file_path=self.data_transformation_config.Y_TRAIN_TRANSFORM_FILE_PATH,
                y_test_transform_file_path=self.data_transformation_config.Y_TEST_TRANSFORM_FILE_PATH,
                preprocessor_file_path=self.data_transformation_config.PREPROCESSOR_FILE_PATH
            )

            return data_transformation_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)


