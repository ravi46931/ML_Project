import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion() 
            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifacts):
        try:
            data_ingestion = DataTransformation(data_ingestion_artifacts, self.data_transformation_config)
            data_transformation_artifacts = data_ingestion.initiate_data_transformation() 
            return data_transformation_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def start_model_trainer(self, data_transformation_artifacts):
        try:
            data_ingestion = ModelTrainer(data_transformation_artifacts, self.model_trainer_config)
            data_ingestion_artifacts = data_ingestion.initiate_model_training() 
            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, data_transformation_artifacts, model_trainer_artifact):
        try:
            model_evaluation = ModelEvaluation(data_transformation_artifacts, model_trainer_artifact)
            model_evaluation.initiate_model_evaluation()
            return None
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifacts)
            self.start_model_evaluation(data_transformation_artifacts, model_trainer_artifact)
        except Exception as e:
            raise CustomException(e, sys)