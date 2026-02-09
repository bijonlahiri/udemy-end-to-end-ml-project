import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f'Numerical Columns: {numerical_columns}')
            logging.info(f'Categorical Columns: {categorical_columns}')

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read Train and Test data')
            logging.info('Obtaining preprocessor object')

            preprocessing_obj = self.get_data_transformation_obj()

            target_column = 'math_score'

            feature_train_df = train_df.drop(target_column, axis=1)
            target_train_df = train_df[target_column]

            feature_test_df = test_df.drop(target_column, axis=1)
            target_test_df = test_df[target_column]

            logging.info('Applying preprocessing transformation on train dataset and test dataset')

            feature_train_preprocessed_df = preprocessing_obj.fit_transform(feature_train_df)
            feature_test_preprocessed_df = preprocessing_obj.transform(feature_test_df)

            final_train_data = np.c_[feature_train_preprocessed_df, target_train_df]
            final_test_data = np.c_[feature_test_preprocessed_df, target_test_df]

            logging.info('Saving preprocessing object')

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                final_train_data,
                final_test_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)