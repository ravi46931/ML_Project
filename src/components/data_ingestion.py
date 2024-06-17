import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        
   
    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv('Data/train.csv')
            
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_ingestion_config.DATA_FILE_PATH)

            data_ingestion_artifacts=DataIngestionArtifacts(
                data_file_path=self.data_ingestion_config.DATA_FILE_PATH
            )

            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion(DataIngestionConfig())
    obj.initiate_data_ingestion() 
    # 5804