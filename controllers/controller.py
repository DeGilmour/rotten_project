from rotten_project.model.genre_predictor import TestModel
from rotten_project.model.recomender import MovieRecomender
from flask import render_template
from flask import Blueprint, request


alfredo = Blueprint('alfredos_two_cents', __name__)


@alfredo.route('/filmology')
def alfredos_two_cents():
    return render_template('Filmology.html',  title="Filmology")


@alfredo.route('/get-prediction', methods=['POST'])
def get_alfredos_two_cents():
    data = request.form
    movie_title = data.get("movie_title", '')
    movie_info = data.get('movie_info')
    model_precit = TestModel(title=movie_title, description=movie_info)
    prediction = model_precit.predict_new_movie()
    prediction = clean_prediction(prediction)
    return {"prediction": "The movie {} seens to be {}".format(movie_title, prediction)}

@alfredo.route('/get-recomendation', methods=['POST'])
def get_alredos_recomendation():
    data = request.form
    movie_title = data.get("movie_title", '')
    movie_info = data.get('movie_info')
    model_recomender = MovieRecomender()
    return {"prediction": "The movie {} seens to be {}".format(movie_title, movie_title)}



def clean_prediction(prediction: list):
    if len(prediction) > 1:
        raise Exception(prediction)
    prediction = prediction[0]
    return ", ".join(prediction)

