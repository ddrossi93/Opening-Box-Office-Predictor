# Opening Weekend Box Office Predictor

This project aims to create an accurate predictor for opening weekends for feature films using machine learning.

## Live Site

https://box-office-predictor.herokuapp.com/

## Features

The "finished" product will have the following features:

- A user-friendly frontend where a user can navigate the webpage and enter the movie they wish to predict.
- Provide an accurate prediction for upcoming films. 
- A continously learning machine learning backend that retrains itself each week based on new box office results.

### Dependencies

- beautifulsoup4
- Flask
- matplotlib
- numpy
- pandas
- requests
- scikit-learn
- statsmodels

### Usage

1. Open site
2. Enter desired movie into form and hit enter
3. Await prediction

### Limitations

Our current model has some limitations that affect the accuracy of the predicions. 

The current model relies heavily on the film's budget, therefore low budget films will have a low prediction, which is not always the case (see [It (2017)](https://www.imdb.com/title/tt1396484/?ref_=fn_al_tt_1)). More variables will be analyzed and considered for the live model in order to continiously improve our predictions. 

### Team Members

- Ashwini Devkota
- Dylan Rossi
- Heather Solis
- Joe Barcus
- Vivian Plasencia
