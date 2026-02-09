import os
import sys
import pickle
from src.logger import logging
from src.exceptions import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

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

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict, random_state=10):
    logging.info(f"Models to evaluate: {list(models.keys())}")
    try:
        final_report = []
        for name, model in models.items():
            logging.info(f'Evaluating for {name}...')
            model_report = {}
            random_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=params[name],
                n_iter=10,
                n_jobs=-1,
                refit=True,
                cv=5,
                verbose=3,
                scoring='r2',
                random_state=random_state
            )
            random_cv.fit(X_train, y_train)
            model_report['Name'] = name
            model_report['Best Params'] = random_cv.best_params_
            model_report['Train R2 Score'] = r2_score(y_train, random_cv.predict(X_train))
            model_report['Test R2 Score'] = r2_score(y_test, random_cv.predict(X_test))
            model_report['Model'] = random_cv
            final_report.append(model_report)

        return final_report
    except Exception as e:
        raise CustomException(e, sys)