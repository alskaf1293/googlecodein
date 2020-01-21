import pandas as pd
import quandl
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense
from tensorflow.keras import Sequential

quandl.ApiConfig.api_key = 'KpHdsREW57Zp6a9ymrE_'
df = quandl.get('EOD/MSFT')


df = df[['Adj_Volume','Adj_Open','Adj_High','Adj_Low','Adj_Close']]
df['PCT_Change'] = ((df['Adj_Close']-df['Adj_Open'])/ df['Adj_Open']) *100
df['PCT_Change_Low'] = ((df['Adj_Low']-df['Adj_Open'])/ df['Adj_Open']) *100
df['PCT_Change_High'] = ((df['Adj_High']-df['Adj_Open'])/ df['Adj_Open']) *100
#df['Future_Close'] = df['Adj_Close'].shift(periods=-1)
#df.dropna(inplace=True)
df['Future_Adj_Close'] = df['Adj_Close'].shift(-1)
df.dropna(inplace=True)

df['Future_PCT_Change'] = (df['Future_Adj_Close']-df['Adj_Close'])/df['Adj_Close']
df = df[['PCT_Change','PCT_Change_Low','PCT_Change_High','Future_PCT_Change']]

X = df[['PCT_Change','PCT_Change_Low','PCT_Change_High']]
y = df[['Future_PCT_Change']]
#df = df[['Adj_Close','PCT_Change','PCT_Change_Low','PCT_Change_High','Future']]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = Sequential()
model.add(Dense(32,activation = 'relu',input_shape=(3,)))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1))
opt = tf.keras.optimizers.Adam(lr=0.03)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])
