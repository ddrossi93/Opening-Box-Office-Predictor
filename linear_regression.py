# -*- coding: utf-8 -*-

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
dataset = pd.read_csv('cleaned_data.csv')




X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# test significance of variables
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((186, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, :]
SL = 0.1

X_Modeled = backwardElimination(X_opt, SL)

# refit model using optimal x variables
from sklearn.cross_validation import train_test_split
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_Modeled, y, test_size = 0.1, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
opt_regressor = LinearRegression()
opt_regressor.fit(X_opt_train, y_opt_train)

# Predicting the Test set results
opt_y_pred = opt_regressor.predict(X_opt_test)

opt_regressor.score(X_opt_test, y_opt_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_opt_test,opt_y_pred)


# plot residuals
plt.scatter(opt_y_pred, opt_y_pred - y_opt_test)
plt.hlines(y=0, xmin=opt_y_pred.min(), xmax=opt_y_pred.max())
plt.show()