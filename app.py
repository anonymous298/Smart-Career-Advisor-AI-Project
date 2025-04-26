# Importing Necessary Dependencies
import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.pipe.prediction_pipeline import PredictionPipeline

# Building our App Backend

app = Flask(__name__)
CORS(app)  # To allow requests from different origins (your HTML page)

# Creating our home route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/career-advisor-ai')
def advisor_ai():
    return render_template('index.html')

@app.route('/submit', methods=["GET", "POST"])
def submit():

    answers_raw = request.args.get('answers', '{}')
    answers = json.loads(answers_raw)

    # Updating Answers Dictonary
    answers = {key : val.lower() for key, val in answers.items()}

    # Here you can do whatever you like with `answers`
    # e.g. run your career model, pick a suggestion, etc.

    # Creating our Prediciton Pipeline Object assigned with all inputs send from UI
    prediction_pipeline = PredictionPipeline(
        Interest=answers['Q1'],
        Skill_Level=answers['Q2'],
        Work_Style=answers['Q3'],
        Education=answers['Q4'],
        Personality=answers['Q5'],
        Problem_Solving=answers['Q6'],
        Tech=answers['Q7'],
        Communication=answers['Q8'],
        Leadership=answers['Q9'],
        Creativity=answers['Q10']
    )

    # Predicting our Output
    prediction = prediction_pipeline.predict()[0]

    print(prediction)

    # print(answers)

    # Then render a result page:
    return render_template('result.html', prediction=prediction)


# @app.route('/submit')
# def submit():
#     answers_raw = request.args.get('answers', '{}')
#     answers = json.loads(answers_raw)

#     # print(answers)

#     # Here you can do whatever you like with `answers`
#     # e.g. run your career model, pick a suggestion, etc.

#     # Then render a result page:
#     return render_template('result.html', answers=answers)

# running our app
if __name__ == '__main__':
    app.run(debug=True, port=8080)