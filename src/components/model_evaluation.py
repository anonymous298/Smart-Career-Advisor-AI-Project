# Importing Necessary Dependencies
import os
import sys

import numpy as np

import mlflow

from tensorflow.keras.models import load_model

from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Creating Logger Object
logger = get_logger('Model_Evaluation')

# Creating our main Class
class ModelEvaluation:
    def __init__(self, model_path):
        self.model_path = model_path

    def evaluate(self, X_train, X_test, y_train, y_test):
        '''
        This method evaluates the model from give model path on training and testing data.

        Parameters:
            X_train, y_train: Training Data.
            X_test, y_test: Testing Data.

        Returns:
            Training-Accuracy.
            Testing-Accuracy.
        '''

        try:

            if os.path.exists(self.model_path):
                logger.info('Evaluating Our Model')

                logger.info('Loading Model From Path')
                model = load_model(self.model_path)
                logger.info('Model Loaded')

                # Evaluating Model On Training Data.
                logger.info("Evaluating Model On Training Data")
                training_loss, training_accuracy = model.evaluate(X_train, y_train)
                
                # Evaluating Model On Testing Data.
                logger.info("Evaluating Model On Testing Data")
                testing_loss, testing_accuracy = model.evaluate(X_test, y_test)

                # Experiment Tracking 
                logger.info('Experiment Tracking Started')

                mlflow.set_experiment('Smart-Career-Advisor-AI')

                with mlflow.start_run(run_name='Model-V0.1'):
                    # Logging Training Loss, Accuracy
                    mlflow.log_metric('Training Loss', training_loss)
                    mlflow.log_metric('Training Accuracy', training_accuracy)

                    # Logging Testing Loss, Accuracy
                    mlflow.log_metric('Testing Loss', testing_loss)
                    mlflow.log_metric('Testing Accuracy', testing_accuracy)

                    # Logging our model
                    mlflow.tensorflow.log_model(model, 'model', registered_model_name='Smart-Career-Advisor-Model-0.1')

                logger.info('Model Evaluation Completed')

                # Returning Accuracies
                return (
                    training_accuracy,
                    testing_accuracy
                )
        
            else:
                logger.error('Model Does Not Exists')

        except Exception as e:
            logger.error(e)
            raise CustomException(e,sys)
        


# For testing purpose
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_preprocessing import DataPreprocessing
    from src.components.model_training import ModelTraining

    data_ingestion = DataIngestion()

    data_path = data_ingestion.load_data()

    data_preprocessing = DataPreprocessing()

    X_train, X_test, y_train, y_test = data_preprocessing.initiate_Preprocessing(data_path=data_path)

    model_training = ModelTraining()

    model_path = model_training.start_training(X_train, X_test, y_train, y_test)

    model_evaluation = ModelEvaluation(model_path=model_path)

    training_acc, testing_acc = model_evaluation.evaluate(X_train, X_test, y_train, y_test)

    print(f'Training Accuracy: {training_acc}')
    print(f'Testing Accuracy: {testing_acc}')
