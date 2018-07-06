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


# test significance of variables
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((2819, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, :]
SL = 0.1

""" After getting optimal x variables """

X_Modeled = backwardElimination(X_opt, SL)

# refit model using optimal x variables
from sklearn.cross_validation import train_test_split
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_Modeled, y, test_size = 0.1, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_opt_train)
poly_reg.fit(X_poly, y_opt_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_opt_train)

poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_opt_test))

lin_reg_2.score(poly_reg.fit_transform(X_opt_test),y_opt_test)

# best score: 0.3807571885792017

from sklearn.metrics import mean_squared_error
mean_squared_error(y_opt_test,poly_pred)