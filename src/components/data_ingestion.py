# Importing Necessary Dependencies
import os
import sys

import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger('Data_Ingestion')

# creating a data class for data path
@dataclass
class DataPath:
    data_path: str = r'notebooks\data\new_career_dataset.csv'
    artifact_data_path = 'artifacts/data.csv'

# Main Data Ingestion Class
class DataIngestion:
    #creating a constructor to call our dataclass
    def __init__(self):
        self.data_path = DataPath()

    def create_artifact(self, data):
        '''
        This method will take loaded dataframe and make a artifact directory and stores that data.

        Parameters:
            data: CSV DataFrame.

        Returns:
            Creates and stores data into artifacts directory
        '''

        try:
            # Creating our artifacts directory if not exists
            logger.info('Creating an artifacts folder')

            os.makedirs(os.path.dirname(self.data_path.artifact_data_path), exist_ok=True)

            # Saving our data
            logger.info('Saving our dataframe to artifacts folder')

            data.to_csv(self.data_path.artifact_data_path, index=False)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def load_data(self):
        '''
        This method will load the data from the path, creates the artifact folder, stores the data and return its path.

        Parameters:
            None.

        Returns:
            data artifacts folder path.
        '''

        try:
            logger.info('Starting the data loading process')

            # Loading CSV Data from path
            if os.path.exists(self.data_path.data_path):

                logger.info('Data Exists in path')
                df = pd.read_csv(self.data_path.data_path)
                logger.info('Data Loaded')

                # Creating artifacts folder and saving data into it
                logger.info('Entering create_artifact method')
                self.create_artifact(data=df)
                logger.info('Exited create_artifact method')

                logger.info('Data Loaded and saved')

                # Returning our saved data path.
                return self.data_path.artifact_data_path

            else:
                logger.warning("Path Not Exists")

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


# For Checking purpose
if __name__ == "__main__":
    data_ingestion = DataIngestion()

    data_path = data_ingestion.load_data()
    print(data_path)