import requests as req
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def scrape(movie):
    print('in scrape')
    movie_data = {
        'BUDGET' : 0,
        'YEAR' : 0,
        'RDJ' : 0
    }

    omdb = 'http://www.omdbapi.com/?apikey=a74304ec&t='
    search = omdb + movie
    results = req.get(search).json()
    poster = results['Poster']
    year = results['Released']
    year = year.split(' ')
    year = year[2]
    print('got year')
    movie_data['YEAR'] = int(year) - 2000
    actors = results['Actors']
    actors = actors.split(',')
    for actor in actors:
        if 'Robert Downey Jr.' in actor:
            movie_data['RDJ'] = 1
            break
    print('got RDJ')

    response = req.get('https://api.themoviedb.org/3/search/movie?api_key=c3bed3ac369b7af1cee6ce88624711e9&language=en-US&query=' + movie + '&page=1&include_adult=false').json()
    id = response['results'][0]['id']
    search = 'https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=c3bed3ac369b7af1cee6ce88624711e9&language=en-US'
    response = req.get(search).json()
    budget = response['budget']
    print(budget)
    movie_data['BUDGET'] = int(budget)

    return movie_data, poster

def fit():
    # Multiple Linear Regression
    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.formula.api as sm

    def backwardElimination(x, sl):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
        regressor_OLS.summary()
        return x

    # Importing the dataset
    dataset = pd.read_csv('eh.csv')

    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values


    # test significance of variables
    import statsmodels.formula.api as sm
    X = np.append(arr = np.ones((175, 1)).astype(int), values = X, axis = 1)

    X_opt = X[:, :]
    SL = 0.05

    """ After getting optimal x variables """

    X_Modeled = backwardElimination(X_opt, SL)
    X_Modeled = X_Modeled[:, 1:]

    # refit model using optimal x variables
    from sklearn.cross_validation import train_test_split
    X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_Modeled, y, test_size = 0.1, random_state = 0)

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 3)
    X_poly = poly_reg.fit_transform(X_opt_train)
    poly_reg.fit(X_poly, y_opt_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_opt_train)

    return poly_reg, lin_reg_2


def predict(movie_data):
    poly, regressor = fit()
    pred_data = []
    pred_data.append(movie_data['BUDGET'])
    pred_data.append(movie_data['YEAR'])
    pred_data.append(movie_data['RDJ'])
    print(pred_data)
    pred_data = [pred_data]
    prediction = regressor.predict(poly.fit_transform(pred_data))

    return prediction
