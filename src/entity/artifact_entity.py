
from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    data_file_path: str
    # train_data_file_path: str
    # test_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    x_train_transform_file_path: str
    x_test_transform_file_path: str
    y_train_transform_file_path: str
    y_test_transform_file_path: str
    preprocessor_file_path: str



@dataclass
class ModelTrainerArtifacts:
    model_file_path: str
    metrics_file_path: str