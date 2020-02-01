# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:15:03 2019

@author: ryanc
"""
#polynomical regression


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#DONT% DO THIS< DATASET TOO SMALL
# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polynomical regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#plotting the linear regression result
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show

#plotting the polynomial regression results
x_grid = np.arange(min(X), max(X),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show

#predicting a new result with the linear model
lin_reg.predict(13/2)

#predicting a new result with the polynomial model
lin_reg_2.predict(poly_reg.fit_transform(X))









