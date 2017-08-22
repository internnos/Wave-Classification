# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:57:38 2017

@author: Amajid Sinar
"""

from keras.models import load_model
import pandas as pd
import numpy as np

model = load_model('FCN.h5')

predict = pd.read_csv("dataset/test-set.csv", delimiter=",", header=None)
predict = predict.iloc[:,1:].values

predict = np.reshape(predict,(predict.shape[0],predict.shape[1],1))


x = model.predict(predict)