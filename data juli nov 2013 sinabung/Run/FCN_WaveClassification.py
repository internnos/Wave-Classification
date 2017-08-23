# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:39:47 2017

@author: Amajid Sinar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

import random

random.seed(5)

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
classifier = Sequential()

#Convolution and MaxPooling
classifier.add(Conv1D(filters=128, kernel_size=8, activation='relu', strides=1, input_shape=(X_train.shape[1],1),
                                                                                            border_mode='same'))
classifier.add(Dropout(0.15))
classifier.add(BatchNormalization())

classifier.add(Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', border_mode='same'))
classifier.add(Dropout(0.15))
classifier.add(BatchNormalization())

classifier.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', border_mode='same'))
classifier.add(Dropout(0.15))
classifier.add(BatchNormalization())

classifier.add(GlobalAveragePooling1D())

#Full Connection
classifier.add(Dense(1,activation='sigmoid'))

#Print summary
print(classifier.summary())

#Configure the learning process
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

earlyStopping=EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.2, patience=5, min_lr=0.0001)

history=classifier.fit(X_train,y_train, batch_size=batch_size, epochs=1000
                       , validation_data=(X_test,y_test), callbacks=[reduce_lr, earlyStopping])


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png', bbox_inches='tight')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.png', bbox_inches='tight')
plt.show()

#---------------------------------------------------------------------------------

prob = classifier.predict_proba(X_test)

np.savetxt("prob.csv",prob,fmt="%s",delimiter=",")

#---------------------------------------------------------------------------------

# serialize model to JSON

