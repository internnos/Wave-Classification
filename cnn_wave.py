# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:08:26 2017

@author: Amajid Sinar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Import training set
training_set = pd.read_csv("training_set.csv", delimiter=";")
X_train = training_set.iloc[:,1:].values
y_train = training_set.iloc[:,0:1].values

#Import test set
test_set = pd.read_csv("test_set_v2.csv", delimiter=";")
X_test = test_set.iloc[:,1:].values
y_test = test_set.iloc[:,0:1].values


#Import test set
#test_set = pd.read_csv("test_set.csv", delimiter=";")
#X_test = test_set.iloc[:,1:].values
#y_test = test_set.iloc[:,0:1].values

#Scale the data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#Convert X into 3D tensor
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Importing the CNN libraries
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Flatten
from keras.layers import Dropout,Dense
from keras.layers.normalization import BatchNormalization

#Initializing the CNN
classifier = Sequential()

#Convolution and MaxPooling
classifier.add(Conv1D(filters=4,kernel_size=4,activation='relu',input_shape=(X_train.shape[1],1)))
classifier.add(MaxPooling1D(strides=4))
#classifier.add(BatchNormalization())

#Flatten
classifier.add(Flatten())

#Full Connection
#classifier.add(Dropout(0.25))
classifier.add(Dense(8, activation='relu'))
#classifier.add(Dropout(0.25))
classifier.add(Dense(1,activation='sigmoid'))

#Print summary
print(classifier.summary())

#Configure the learning process
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=classifier.fit(X_train,y_train, batch_size=64, epochs=50, validation_data=(X_test,y_test))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------
#Evaluating
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Conv1D(filters=4,kernel_size=4,activation='relu',input_shape=(X_train.shape[1],1)))
    classifier.add(MaxPooling1D(strides=4))
    classifier.add(BatchNormalization())
    classifier.add(Flatten())    
    classifier.add(Dropout(0.25))
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(1,activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 32, epochs = 5)
accuracies = cross_val_score(classifier, X=X_train, y=y_train, cv=10)

mean = accuracies.mean()
variance = accuracies.std()    
#---------------------------------------------------------------------------------
#Parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer, hnode, dropout1=0, dropout2=0, filters, kernel_size,strides):
    classifier = Sequential()
    classifier.add(Conv1D(filters=filters,kernel_size=kernel_size,activation='relu',input_shape=(X_train.shape[1],1)))
    classifier.add(MaxPooling1D(strides=strides))
    #classifier.add(BatchNormalization())    
    classifier.add(Flatten())
    classifier.add(Dropout(dropout1))
    classifier.add(Dense(hnode, activation='relu'))
    classifier.add(Dropout(dropout2))
    classifier.add(Dense(1,activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25,32],
              'epochs': [5,10],
              'optimizer': ['adam', 'rmsprop'],
              'dropout1' : [0.1,0.15,0.2,0.25],
              'dropout2' : [0.1,0.15,0.2,0.25],
              'hnode' : [6,7,8,9,10],
              'filters': [2,3,4,5,6],
              'kernel_size': [2,3,4,5,6],
              'strides': [2,3,4,5,6]
              
              }
grid_search = GridSearchCV(estimator=classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
#---------------------------------------------------------------------------------
performace = classifier.evaluate(X_test,y_test)

y_pred = classifier.predict_classes(X_test)


    
    

