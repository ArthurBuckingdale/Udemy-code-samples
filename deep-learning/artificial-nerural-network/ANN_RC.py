# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:05:05 2019

@author: ryanc
"""

#Artificail Neural Network

#Part 1 data preprocession

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#part 2 making the ANN

#importing the keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#adding the second hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error', metrics = ['cosine_proximity'] )

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=100 , epochs = 100)

#part 3 Making the predictions and evaluating
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




























