"""
Created on Tue Feb  6 18:42:29 2024
As found on: https://blog.gopenai.com/predicting-stock-prices-with-lstm-and-gru-a-step-by-step-guide-381ec1554edf
@author: u0099498
"""
#Comment this out to run on GPU instead of CPU (CPU faster for smaller model sizes due to GPU overhead and smaller clock speed)
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from matplotlib import pyplot as plt


import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, InputLayer, Reshape, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import save_model
import keras_tuner as kt

import xgboost as xgb
from scipy.stats import binom
from decimal import Decimal
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pickle
import helpers


#Load data from pickle file (files defined in functions in helpers)
#x, y, dates, tckr  = helpers.loaddatals()
#x, y, dates, tckr = helpers.loaddata()

with open("datals-2019-2024", 'rb') as f:
#with open("data-2011-2024", 'rb') as f:
       data0 = pickle.load(f)
x = data0['x']; y = data0['y']; dates = data0['dates']; tckr = data0['tckr']

#Remove highly anomalous days with market crashes (more than 50 stocks which fall more than 5% pre-market)
indices = np.array([])
unique, counts = np.unique(dates, return_counts=True)
date2rem = unique[np.where(counts>50)]
for i in range(len(date2rem)):
    indices = np.concatenate((indices,np.argwhere(dates == date2rem[i]).ravel()))
#x = np.delete(x,indices.astype(int),axis=0); y = np.delete(y,indices.astype(int));

#Shuffle data
#indices = np.arange(len(y))
#np.random.shuffle(indices)
#x = x[indices]; y = y[indices]

y = np.array([0 if yi < 0 else 1 for yi in y])

split_index1 = int(len(y)*0.9)
x1 = x[:split_index1]; y1 = y[:split_index1]

#Shuffle data
indices = np.arange(len(y1))
np.random.shuffle(indices)
x1 = x1[indices]; y1 = y1[indices]

split_index2 = int(len(y1)*0.8)

data_x_train = x1[:split_index2]
data_x_val =  x1[split_index2:]
data_y_train = y1[:split_index2]
data_y_val = y1[split_index2:]
data_x_test = x[split_index1:]
data_y_test = y[split_index1:]

# normalize
#sc_x = MinMaxScaler(feature_range=(0, 1))
sc_y = MinMaxScaler(feature_range=(0, 1))
sc_x = StandardScaler()
#sc_y = StandardScaler()
data_x_train = sc_x.fit_transform(data_x_train)
data_x_val = sc_x.transform(data_x_val)
data_x_test = sc_x.transform(data_x_test)
data_y_train = sc_y.fit_transform(data_y_train.reshape(-1,1))
data_y_val = sc_y.transform(data_y_val.reshape(-1,1))
data_y_test = sc_y.transform(data_y_test.reshape(-1,1))

#Predicting model
#model = helpers.getmodel(1,x.shape[1])

def build_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=(data_x_train.shape[1],1)))
    for i in range(hp.Int('LSTM_layers', 0, 3)):
        model.add(LSTM(hp.Choice(f'LSTM_layer_{i}', values=[16,32,64,128,256]),return_sequences=True))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(LSTM(hp.Choice('last_LSTM', values=[16,32,64,128,256])))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    for i in range(hp.Int('Dense_layers', 0, 4)):
        model.add(Dense(hp.Choice(f'Dense_layer_{i}', values=[16,32,64,128,256])))
    #model.add(Dense(1))
    model.add(Dense(8, activation="softmax"))
    model.add(Dense(units=1,activation="sigmoid"))
    initial_learning_rate = hp.Choice('Learning rate', values=[1e-2, 1e-3, 1e-4])
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
       initial_learning_rate,
       decay_steps=len(y1)/50,
       decay_rate=0.96,
       staircase=True)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='mse', optimizer=opt)
    return model

#tuner = kt.Hyperband(build_model,
#                     objective='val_accuracy',
#                     max_epochs=50,
#                     factor=3,
#                     overwrite=True,
#                     project_name='datals5m')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

#tuner.search(data_x_train, data_y_train, epochs=100, batch_size=50, callbacks=[stop_early], 
#             validation_data=(data_x_val,data_y_val))

# Get the optimal hyperparameters
#best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = Sequential()
model.add(InputLayer(input_shape=(data_x_train.shape[1],1)))
#model.add(LSTM(256, return_sequences=True)); model.add(Dropout(0.3))
model.add(LSTM(32)); model.add(Dropout(0.3))
model.add(Dense(16))
#model.add(Dense(8, activation="softmax"))
model.add(Dense(units=1,activation="sigmoid"))
#model.add(Dense(units=1))

callback = EarlyStopping(monitor='val_accuracy',patience=20)
#callback = EarlyStopping(monitor='val_loss',patience=10)

initial_learning_rate = 1e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
       initial_learning_rate,
       decay_steps=len(y1)/50,
       decay_rate=0.96,
       staircase=True)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
#model.compile(optimizer=opt,loss='mse')

model.compile(optimizer=opt,
              loss='binary_crossentropy',
             metrics = ['accuracy'])

#model = tuner.hypermodel.build(best_hps)

history = model.fit(data_x_train,data_y_train, batch_size=50, epochs=100, validation_data=(data_x_val,data_y_val),
          callbacks=[callback], shuffle=True)

#val_loss_per_epoch = history.history['val_loss']
#best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

# Retrain the model
#model = tuner.hypermodel.build(best_hps)
#history = model.fit(data_x_train,data_y_train, batch_size=5, epochs=best_epoch, #validation_data=(data_x_val,data_y_val))

#model = xgb.XGBRegressor(n_estimators=5000)
#model.fit(data_x_train, data_y_train,
#        eval_set=[(data_x_train, data_y_train), (data_x_val, data_y_val)],
#        early_stopping_rounds=200,
#       verbose=False)

predicted_val = model.predict(data_x_val)
predicted_val = sc_y.inverse_transform(predicted_val.reshape(-1,1))
predicted_train = model.predict(data_x_train)
predicted_train = sc_y.inverse_transform(predicted_train.reshape(-1,1))
predicted_test = model.predict(data_x_test)
predicted_test = sc_y.inverse_transform(predicted_test.reshape(-1,1))

data_x_train = sc_x.inverse_transform(data_x_train)
data_x_val = sc_x.inverse_transform(data_x_val)
data_x_test = sc_x.inverse_transform(data_x_test)
data_y_train = sc_y.inverse_transform(data_y_train)
data_y_val = sc_y.inverse_transform(data_y_val)
data_y_test = sc_y.inverse_transform(data_y_test)

print("For training data:")
sucrate = [1 if np.sign(data_y_train[i]) ==\
                np.sign(predicted_train[i]) else 0 for i in range(len(predicted_train))]
binom_dist = binom(len(predicted_train), 0.5)
prob = binom_dist.pmf(np.sum(sucrate))
print("The success rate of predicting higher or lower price is: %.2E"% Decimal(np.sum(sucrate)/len(sucrate)))
print("The chance of this happening through random guessing is: %.2E, which is %.2E sigma away from expected value"\
      % (Decimal(prob),Decimal(abs(len(sucrate)/2-np.sum(sucrate))/binom_dist.std())))
sucrate = [1 if np.sign(data_y_val[i]) ==\
                np.sign(predicted_val[i]) else 0 for i in range(len(predicted_val))]
binom_dist = binom(len(predicted_val), 0.5)
prob = binom_dist.pmf(np.sum(sucrate))
print("For validation data:")
print("The success rate of predicting higher or lower price is: %.2E"% Decimal(np.sum(sucrate)/len(sucrate)))
print("The chance of this happening through random guessing is: %.2E, which is %.2E sigma away from expected value"\
      % (Decimal(prob),Decimal(abs(len(sucrate)/2-np.sum(sucrate))/binom_dist.std())))
print("For test data:")
sucrate = [1 if np.sign(data_y_test[i]-0.5) ==\
                np.sign(predicted_test[i]-0.5) else 0 for i in range(len(predicted_test))]
#sucrate = [1 if np.sign(data_y_test[i]) ==\
#                np.sign(predicted_test[i]) else 0 for i in range(len(predicted_test))]
binom_dist = binom(len(predicted_test), 0.5)
prob = binom_dist.pmf(np.sum(sucrate))
print("The success rate of predicting higher or lower price is: %.2E"% Decimal(np.sum(sucrate)/len(sucrate)))
print("The chance of this happening through random guessing is: %.2E, which is %.2E sigma away from expected value"\
      % (Decimal(prob),Decimal(abs(len(sucrate)/2-np.sum(sucrate))/binom_dist.std())))

model.save('my_model.keras')

plt.plot(np.concatenate((data_y_train,data_y_val,data_y_test)),label='Real change')
plt.plot(np.concatenate((predicted_train,predicted_val,predicted_test)),label='Predicted change')
plt.axvline(x=len(data_y_train),linewidth=3,color='r')
plt.axvline(x=len(y1),linewidth=3,color='r')
plt.legend()
helpers.plotlearn(history,type='loss')
plt.show()