# Importing Necessary Dependencies
import os
import sys

import pandas as pd
import numpy as np

import pickle

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from tensorflow.keras.models import load_model

# Creating Logger Object
logger = get_logger('Prediction_Pipeline')

# Creating our dataclass for paths
@dataclass
class Paths:
    X_preprocessors_path: str = os.path.join('models', 'X_preprocessors.pkl')
    y_preprocessor_path: str = os.path.join('models', 'y_preprocessor.pkl')
    model_path: str = os.path.join('models', 'trained_model.h5')

# Creating our main class
class PredictionPipeline:
    def __init__(self,
                 Interest,
                 Skill_Level,
                 Work_Style,
                 Education,
                 Personality,
                 Problem_Solving,
                 Tech,
                 Communication,
                 Leadership,
                 Creativity):
        
        self.paths = Paths()

        self.Interest = Interest
        self.Skill_Level = Skill_Level
        self.Work_Style = Work_Style
        self.Education = Education
        self.Personality = Personality
        self.Problem_Solving = Problem_Solving
        self.Tech = Tech
        self.Communication = Communication
        self.Leadership = Leadership
        self.Creativity = Creativity

    def get_models(self):
        '''
        This method will load all the models and returns it.

        Returns:
            X_preprocessors, y_preprocessor, model
        '''

        try:
            # Loading our X_Preprocessors
            with open(self.paths.X_preprocessors_path, 'rb') as file:
                X_preprocessors = pickle.load(file)

            # Loading our y_preprocessor
            with open(self.paths.y_preprocessor_path, 'rb') as file:
                y_preprocessor = pickle.load(file)

            # Loading our Trained Model
            model = load_model(self.paths.model_path)

            # Returning our objects

            return (
                X_preprocessors,
                y_preprocessor,
                model
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        '''
        This method will convert all the columns to a dataframe.

        Returns:
            DataFrame
        '''
        try:

            # Converting all inputs into a Dictionary 
            custom_data_input_dict = {
                'Interest' : [self.Interest],
                'Skill_Level' : [self.Skill_Level],
                'Work_Style' : [self.Work_Style],
                'Education' : [self.Education],
                'Personality' : [self.Personality],
                'Problem_Solving' : [self.Problem_Solving],
                'Tech' : [self.Tech],
                'Communication' : [self.Communication],
                'Leadership' : [self.Leadership],
                'Creativity' : [self.Creativity]
            }

            # Converting our inputs dictonary to DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            # returning our dataframe
            return df
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def predict(self):
        '''
        This method will preprocessed the input data and gives it to model for prediction and returns the response.

        Returns:
            model prediction.
        '''

        try:
            logger.info('Prediction Started')

            # Loading our Data Frame
            dataframe = self.get_data_as_dataframe()

            # Loading all model objects
            X_preprocessors, y_preprocessor, model = self.get_models()

            # Preprocessing our dataframe
            input_qurie = np.column_stack([
                X_preprocessors[col].transform(dataframe[col]) for col in dataframe.columns
            ])

            # Predicting our input qurie
            prediction = model.predict(input_qurie)

            pred_class = np.argmax(prediction, axis=1)

            # Decoding our prediction class
            decoded_class = y_preprocessor.inverse_transform(pred_class)

            # Returning our Prediction
            return decoded_class         

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        


# For Testing Purpose
if __name__ == '__main__':
    # prediction_pipeline = PredictionPipeline('technology','advanced','flexible','phd','thinker','logical','high','poor','moderate','very high')
    prediction_pipeline = PredictionPipeline('technology','advanced','team','phd','extrovert','logical','high','fluent','strong','very high')

    prediction = prediction_pipeline.predict()

    print(f'Prediction: {prediction}')