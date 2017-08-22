# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:50:50 2017

@author: Amajid Sinar
"""

import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------
#Merge training set
VTA_training_set = pd.read_csv("dataset/VTA-training-set.csv", delimiter=";", header=None).values
VTB_training_set = pd.read_csv("dataset/VTB-training-set.csv", delimiter=";", header=None).values
training_set = np.concatenate((VTA_training_set,VTB_training_set))

#Merge test set
VTA_test_set = pd.read_csv("dataset/VTA-test-set.csv", delimiter=";", header=None).values
VTB_test_set = pd.read_csv("dataset/VTB-test-set.csv", delimiter=";", header=None).values
test_set = np.concatenate((VTA_test_set,VTB_test_set))

#Create csv
np.savetxt('dataset/training-set.csv',training_set,fmt="%s",delimiter=",")
np.savetxt('dataset/test-set.csv',test_set,fmt="%s",delimiter=",")
