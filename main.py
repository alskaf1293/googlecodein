import pandas as pd
import quandl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm, linear_model
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
import datetime

#In order to consistently get data from Quandl's server, one needs to sign up
#and recieve an API key. Input this as a strng to the API_KEY variable below
API_KEY = ''
quandl.ApiConfig.api_key = API_KEY

#############################################################################

#get information
df = quandl.get('EOD/MSFT')

#variablethat stores how many days into the future you want to project
days_to_project = 1

#preprocessing information
df = df[['Adj_Volume','Adj_Open','Adj_High','Adj_Low','Adj_Close']]
df['PCT_Change'] = ((df['Adj_Close']-df['Adj_Open'])/ df['Adj_Open']) *100
df['PCT_Change_Low'] = ((df['Adj_Low']-df['Adj_Open'])/ df['Adj_Open']) *100
df['PCT_Change_High'] = ((df['Adj_High']-df['Adj_Open'])/ df['Adj_Open']) *100
df['Future_Adj_Close'] = df['Adj_Close'].shift(-days_to_project)
df.dropna(inplace=True)

df = df[['Adj_Close','PCT_Change','PCT_Change_Low','PCT_Change_High','Future_Adj_Close']]

X = df.drop('Future_Adj_Close', axis=1)
y = df[['Future_Adj_Close']]

X = preprocessing.scale(X)
y = preprocessing.scale(y)
y = np.array(y)

#Tensorboard setup
log_dir = "performance/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

#model split train,test, and fitting
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = Sequential()
model.add(Dense(1, input_shape=(4,),activation = 'linear'))
opt = tf.keras.optimizers.Adam(lr=0.03)
loss = tf.keras.losses.MSE
model.compile(optimizer=opt,loss=loss,metrics=['MSE'])
model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test),callbacks=[tensorboard])
