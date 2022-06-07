from rotten_project.model.genre_predictor import TestModel
from rotten_project.model.recomender import MovieRecomender, PreProcessing
from flask import render_template
from flask import Blueprint, request
import pandas as pd

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
    # pre_pro = PreProcessing()
    # cleaned = pre_pro.clean_dataset()
    # cleaned.to_csv('cleaned_rotten_ds.csv')
    movie_recomender = MovieRecomender()
    movie_recomender.find_movies_calculate(movie_title)
    return {"prediction": "The movie {} seens to be ".format(movie_title)}


def clean_prediction(prediction: list):
    if len(prediction) > 1:
        raise Exception(prediction)
    prediction = prediction[0]
    return ", ".join(prediction)
