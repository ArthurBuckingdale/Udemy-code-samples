# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:03:41 2019

@author: ryanc
"""

#this is the recurrent neural network section. Should be tight. We will be making
#an LSTM RNN that will predict the upward or downward trend of stock prices.


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#applying feature scaling to the data
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps
x_train=[]
y_train=[]
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping the data 
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1) )

#second part, assembling the RNN
#importing the libraries from kerqas

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialise the RNN
regressor = Sequential()
 
#adding the first layer and dropout regularisation 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the second LSTM layer with droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the third lstm layer and some dropout regularisation 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding the fourth lstm layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer of the neural network
regressor.add(Dense(units=1)) 

#compiling the RNN
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')

#fitting the RNN to the training set
regressor.fit(x_train,y_train, epochs= 4, batch_size = 32 )

#part three making predictions and visulaise results
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis= 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test=[]
for i in range(60, 80):
    x_test.append(inputs[i-60:i,0])
x_test= np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1) )
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the results
plt.plot(real_stock_price, color = 'blue' , label = 'real google stock price')
plt.plot(predicted_stock_price, color = 'red' , label = 'predicted google stock price')
plt.title('google stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()












































    
