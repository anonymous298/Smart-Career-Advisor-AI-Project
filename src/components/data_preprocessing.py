# Importing Necessary Dependencies
import os
import sys
import pickle

import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# creating our logger object
logger = get_logger('Data_Preprocessing')

# Creating our data class for paths
@dataclass
class DataPaths:
    training_data_path: str = 'artifacts/train.csv'
    testing_data_path: str = 'artifacts/test.csv'

# Creating our main data preprocessing class
class DataPreprocessing:
    def __init__(self):
        self.data_paths = DataPaths()

        self.DEPENDENT_FEATURE = "Career"

    def data_splitter(self, data):
        '''
        This method will take data and convert it into training and testing data and stores it in artifacts folder.

        Parameters:
            data: Dataframe.

        Returns:
            train and test data paths
        '''

        try:
            # Splitting our data into train and test part 80/20
            logger.info('Splitting data into 80/20 part')

            train_data, test_data = train_test_split(data, test_size=0.2)

            logger.info('Data Splitted')
            logger.info('Saving train and test data to artifacts')

            #saving train and test data to artifact
            logger.info('Saving train data to artifact')
            train_data.to_csv(self.data_paths.training_data_path, index=False)

            logger.info('Saving test data to artifacts')
            test_data.to_csv(self.data_paths.testing_data_path, index=False)

            logger.info('train and test data saving done')

            # returning training and testing data paths
            return (
                self.data_paths.training_data_path,
                self.data_paths.testing_data_path
            )
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def training_testing_splitter(self, data):
        '''
        This method will take data and returns X_train, X_test, y_train, y_test

        Parameters:
            data: DataFrame
        
        Returns:
            X_train, X_test, y_train, y_test
        '''

        try:
            # Sending data to split it into train and testing part.
            logger.info('Splitting it into training and testing part')

            train_path, test_path = self.data_splitter(data=data)

            logger.info('Loading train and test data path')

            logger.info('Loading training data path')
            train_data = pd.read_csv(train_path)

            logger.info('Loading testing data path')
            test_data = pd.read_csv(test_path)

            logger.info('Train and Test data loaded')

            # Splitting data into Independent and Dependent features

            logger.info("Splitting training data to dependent and independt feature")
            X_train, y_train = train_data.drop([self.DEPENDENT_FEATURE], axis=1), train_data[self.DEPENDENT_FEATURE]

            logger.info('Splitting testing data into dependent and independent feature')
            X_test, y_test = test_data.drop([self.DEPENDENT_FEATURE], axis=1), test_data[self.DEPENDENT_FEATURE]

            logger.info('Data Splitting Completed')
            
            # Returning X_train, X_test, y_train, y_test
            return (
                X_train, 
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def get_preprocessors(self, X_train, y_train):
        '''
        This method creates X_preprocessor and y_preprocessor and returns it

        Parameters:
            columns: for creating X_preprocessor columns is required for preprocessor.

        Returns:
            X_preprocessor, y_preprocessor.
        '''

        try:
            # Creating our preprocessor object
            logger.info('Creating our preprocessors object')

            logger.info("Creating our X_preprocessor")

            X_preprocessors = {
                col: LabelEncoder().fit(X_train[col]) for col in X_train.columns
            }

            logger.info('Creating our y_preprocessor')
            y_preprocessor = LabelEncoder().fit(y_train)

            logger.info('Preprocessor Object Created')

            # Returning the preprocessor objects
            return (
                X_preprocessors,
                y_preprocessor
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def initiate_Preprocessing(self, data_path):
        '''
        This main method will take data path, splits it, preprocessed it and returns it.

        Parameters:
            data_path: our data path.

        Returns:
            X_train, X_test, y_train, y_test.
        '''

        try:
            # Starting our data preprocessing
            logger.info('Data Preprocessing Initialized')

            if os.path.exists(data_path):
                # reading our data from path
                logger.info("Reading our data")
                data = pd.read_csv(data_path)
                logger.info('Data Loaded')

                # Getting X_train, X_test, y_train, y_test
                X_train, X_test, y_train, y_test = self.training_testing_splitter(data=data)

                # Getting our preprocessor objects
                logger.info('Entering get_preprocessor method')
                X_preprocessors, y_preprocessor = self.get_preprocessors(X_train=X_train, y_train=y_train)

                # Initiating preprocessing on our data
                logger.info("Starts preprocessing")

                logger.info("Preprocessing on Independent Data")

                # Encode X_train
                X_train_trf = np.column_stack([
                    X_preprocessors[col].transform(X_train[col]) for col in X_train.columns
                ])

                # Encode X_test
                X_test_trf = np.column_stack([
                    X_preprocessors[col].transform(X_test[col]) for col in X_test.columns
                ])

                logger.info("Preprocessing on Dependent Features")

                # transforming on y_train
                y_train_trf = y_preprocessor.transform(y_train)

                # Transforming on y_test
                y_test_trf = y_preprocessor.transform(y_test)

                logger.info("All Preprocessing Completed")

                logger.info('Saving our models to model path')

                os.makedirs('models', exist_ok=True)

                # Saving X_preprocessors (dict of LabelEncoders)
                with open("models/X_preprocessors.pkl", "wb") as f:
                    pickle.dump(X_preprocessors, f)

                # Saving y_preprocessor (single LabelEncoder)
                with open("models/y_preprocessor.pkl", "wb") as f:
                    pickle.dump(y_preprocessor, f)

                # Returning Transformed and model ready to train data
                return (
                    X_train_trf,
                    X_test_trf,
                    y_train_trf,
                    y_test_trf
                )
            
            else:
                logger.error('Data Not exists')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


# For testing purpose
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    data_ingestion = DataIngestion()

    data_path = data_ingestion.load_data()

    data_preprocessing = DataPreprocessing()

    X_train, X_test, y_train, y_test = data_preprocessing.initiate_Preprocessing(data_path=data_path)

    print(X_train)