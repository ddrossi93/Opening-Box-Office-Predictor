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
dataset = pd.read_csv('slimmed_data.csv')
# test these and turn into dummy variables later
dataset = dataset.drop('Team', 1)
dataset = dataset.drop('Opponent', 1)

# drop rows with missing data
cleaned_df = dataset.dropna()
# create dummy variables
cleaned_df = pd.get_dummies(cleaned_df)


# few games are played in january. so remove
cleaned_df = cleaned_df.drop('Month_January', 1)

# avoid dummy variable trap
cleaned_df = cleaned_df.drop('Month_December', 1)



X = cleaned_df.iloc[:, 4:27].values
y = cleaned_df.iloc[:, 0].values


import statsmodels.formula.api as sm
X = np.append(arr = np.ones((2819, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, :]
SL = 0.1

regressor_OLS = sm.OLS(endog = y, exog = X_Modeled).fit()
regressor_OLS.summary()

X_Modeled = backwardElimination(X_opt, SL)


""" After getting optimal x variables """

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.1, random_state = 0)


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators' : list(range(1,999,2))}
grid = GridSearchCV(forest, param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_

grid.best_score_

opt_y_pred = grid.predict(X_test)


# best n_estimators: 169
# best score: 0.32784607608348343