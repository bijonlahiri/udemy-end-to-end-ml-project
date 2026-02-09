import os
import sys
import pickle
from src.logger import logging
from src.exceptions import CustomException

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f'Successfully created filepath: {dir_path}')

        with open(filepath, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info('Successfully saved model object')
    except Exception as e:
        raise CustomException(e, sys)