from src.exception import CustomException
from src.logger import logging
import sys
import pickle

def save_object(file_path, object):
     try:
          logging.info(f"Saving object into {file_path}")
          with open(file_path, 'wb') as file:
                pickle.dump(object, file)
                logging.info(f"Object saved into {file_path}")
     except Exception as e:
          raise CustomException(e, sys)


def load_object(file_path):
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as file:
                object = pickle.load(file)
        logging.info(f"Object loaded successfully from {file_path}")
        return object
    except Exception as e:
        raise CustomException(e, sys)