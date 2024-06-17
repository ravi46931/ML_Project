import os  
import sys
import pandas as pd
from joblib import load
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object
from src.constants import MODEL_FOLDER, PREPROCESSOR_FILE, MODEL_FILE

class PredictionPipeline:
    def __init__(self):
        self.model_folder = MODEL_FOLDER
        self.preprocessor_file = PREPROCESSOR_FILE
        self.model_file = MODEL_FILE

    def predict_data(self, input_data):
        try:
            model = load_object(
                file_path=os.path.join(self.model_folder, self.model_file)
            )            
            preprocessor = load(filename=os.path.join(self.model_folder, self.preprocessor_file))
            data = pd.DataFrame([input_data])
            transform_data = preprocessor.transform(data)
            transform_data = pd.DataFrame(transform_data, columns=preprocessor.get_feature_names_out())
            predicted_value = model.predict(transform_data)
            return round(predicted_value[0],2)
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_prediction_pipeline(self, input_data):
        try:
            logging.info(f"Prediction started for {input_data}")
            predicted_value = self.predict_data(input_data)
            logging.info(f"Predicted value: {predicted_value}")
            return predicted_value

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__=="__main__":
    pred_pipeline = PredictionPipeline()
    data = {
        'carat': 1.52,
        'cut': 'Premium', 
        'color': 'F',
        'clarity': 'VS2', 
        'depth': 62.2,   
        'table': 58,
        'x': 9,
        'y': 7.33,
        'z': 4.55
    }
    prediction = pred_pipeline.initiate_prediction_pipeline(input_data=data)
    print(f"Predicted Price of the Diamond: {prediction}")
