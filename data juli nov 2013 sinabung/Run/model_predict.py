# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:57:38 2017

@author: Amajid Sinar
"""

#from keras.models import load_model
import pandas as pd
import numpy as np

training_set = pd.read_csv("dataset/training-set.csv", delimiter=",", header=None)
X_train = training_set.iloc[:,1:].values
y_train = training_set.iloc[:,0:1].values
batch_size = min(X_train.shape[0]/10, 16)

#Import test set
test_set = pd.read_csv("dataset/test-set.csv", delimiter=",", header=None)
X_test = test_set.iloc[:,1:].values
y_test = test_set.iloc[:,0:1].values

X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)

X_train = (X_train - X_train_mean)/(X_train_std)
X_test = (X_test - X_train_mean) / (X_train_std)

#Convert X into 3D tensor
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Importing the CNN libraries
from keras.models import Sequential
from keras.layers import Conv1D,GlobalAveragePooling1D
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization

#FCN
model = Sequential()

#Convolution and MaxPooling
model.add(Conv1D(filters=128, kernel_size=8, activation='relu', strides=1, input_shape=(X_train.shape[1],1),
                                                                                            border_mode='same'))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', border_mode='same'))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', border_mode='same'))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(GlobalAveragePooling1D())

#Full Connection
model.add(Dense(1,activation='sigmoid'))

#Print summary
print(model.summary())

#Configure the learning process
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Loaded model from disk")
fname = "fcn-model.hdf5"
model.load_weights(fname)
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))