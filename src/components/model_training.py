# Importing Necessary Dependencies
import os
import sys

import numpy as np

from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.utils.exception import CustomException
from src.utils.logger import get_logger

# Creating our Logger object
logger = get_logger('Model_Training')

# creating our data class for model path
@dataclass
class ModelPaths:
    model_path: str = 'models/trained_model.h5'

# Main Model training class
class ModelTraining:
    def __init__(self):
        self.model_path = ModelPaths()

    def model_building(self, input_dim, output_neurons):
        '''
        This method will build a neural network architecture and returns it for training purpose.

        Parameters:
            input_shape: input shape for neural network input layer.
            output_neurons: output neurons for neural network output layer neurons.

        Returns: 
            model object.
        '''

        try: 
            logger.info('Building our Neural Network Architecture')

            # Building Neural Network Model Architecture.
            model = Sequential()

            # Adding first Dense Layer
            model.add(Dense(256, activation='relu', input_shape=(input_dim, )))
            # Adding Dropout Layer
            model.add(Dropout(0.3))
            # Adding Second Dense Layer
            model.add(Dense(128, activation='relu'))
            # Adding Dropout Layer
            model.add(Dropout(0.3))
            # Adding Output layer
            model.add(Dense(output_neurons, activation='softmax'))

            logger.info("Compiling Our Model")

            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            logger.info('Model Builded Successfully')
            
            # Returning our model
            return model

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def start_training(self, X_train, X_test, y_train, y_test):
        '''
        This main method start training of our neural network on training data.

        Parameters:
            X_train: Training X data.
            X_Test: Testing X data.
            y_train: Training y data.
            y_test: Testing y data.

        Returns:
            model path for evaluation on next component.
        '''

        try:
            logger.info('Model Training Started')

            logger.info('Getting our model')

            # Extracting X_train feature dim
            input_dim = X_train.shape[1]

            # Extracting y classes size
            output_dim = len(np.unique(y_train))

            # Getting our model
            model = self.model_building(input_dim=input_dim, output_neurons=output_dim)

            # Creating Early Stopping Object For Our Model
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

            logger.info('Model training initiated')

            model.fit(
                X_train, 
                y_train,
                validation_data=(X_test, y_test),
                batch_size=32,
                callbacks=[early_stop],
                epochs=50
            )

            logger.info('Model Trained Successfully')

            logger.info('Saving our model to path')

            if os.path.exists(os.path.dirname(self.model_path.model_path)):
                # Saving model to path
                model.save(self.model_path.model_path)

                logger.info(f'Model saved successfully at {self.model_path.model_path}')

            else:
                logger.error('Models directory does not exists')

            logger.info('Model training related things done successfully')

            return self.model_path.model_path

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)



# For testing purpose
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_preprocessing import DataPreprocessing

    data_ingestion = DataIngestion()

    data_path = data_ingestion.load_data()

    data_preprocessing = DataPreprocessing()

    X_train, X_test, y_train, y_test = data_preprocessing.initiate_Preprocessing(data_path=data_path)

    model_training = ModelTraining()

    model_path = model_training.start_training(X_train, X_test, y_train, y_test)