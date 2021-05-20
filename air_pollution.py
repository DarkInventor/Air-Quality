# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 23:45:02 2019

@author: ktmeh
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('pollution_data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:2])
X[:, 0:2] = imputer.transform(X[:, 0:2])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#graph
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Year vs PM2.5 (Training set)')
plt.xlabel('Years')
plt.ylabel('Level of PM2.5')
plt.show()

