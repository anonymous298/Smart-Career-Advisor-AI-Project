# Importing Necessary Dependencies
import os
import sys

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation

# Creating Logger Object
logger = get_logger('Training_Pipeline')

# Creating our main function
def main():
    '''
    This is our main function which will connect and run the whole components.
    '''

    try:
        logger.info('Training Pipeline Initialized')

        logger.info('Entering Data Ingestion Component')

        # Data Ingestion Component
        data_ingestion = DataIngestion()

        # Extracting Data
        data_path = data_ingestion.load_data()

        logger.info('Entering Data Preprocessing Component')

        # Data Preprocessing Component
        data_preprocessing = DataPreprocessing()

        # Cleaning and getting model ready to train data
        X_train, X_test, y_train, y_test = data_preprocessing.initiate_Preprocessing(data_path=data_path)

        logger.info("Entering Model Training Component")

        # Model Training Component
        model_training = ModelTraining()

        # Training Our Model
        model_path = model_training.start_training(X_train, X_test, y_train, y_test)

        logger.info("Entering Model Evaluation Component")

        # Model Evaluation Component
        model_evaluation = ModelEvaluation(model_path=model_path)

        # Evalating our model
        training_acc, testing_acc = model_evaluation.evaluate(X_train, X_test, y_train, y_test)

        # Printing Accurcies
        print(f"Training Accuracy: {training_acc}")
        print(f"Testing Accuracy: {testing_acc}")

        logger.info('Training Pipeline Completed')

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    main()