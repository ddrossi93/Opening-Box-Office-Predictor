# import dependencies
from flask import Flask, render_template, jsonify, redirect, current_app, request
import movies
import requests


# create instance of Flask
app = Flask(__name__)

# create index route
@app.route("/")
def index():

    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    movie_text = request.form['movie-input']
    movie = movie_text.replace(":","+")
    movie = movie.replace("'","")
    movie = movie.replace(' ', '+')
    movie_data = movies.scrape(movie)
    prediction = movies.predict(movie_data)
    prediction = float(prediction)

    return render_template("prediction.html", movie=movie_text, prediction=prediction, movie_data=movie_data)


if __name__ == "__main__":
    app.run(debug=True)
