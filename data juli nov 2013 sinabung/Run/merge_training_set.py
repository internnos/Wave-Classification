# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:50:50 2017

@author: Amajid Sinar
"""

import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------
#Merge training set
VTA_training_set = pd.read_csv("dataset/VTA_training_set.csv", delimiter=";")
VTB_training_set = pd.read_csv("dataset/VTB_training_set.csv", delimiter=";")
training_set = np.concatenate((VTA_training_set,VTB_training_set))

#Merge test set
VTA_test_set = pd.read_csv("dataset/VTA_test_set.csv", delimiter=";")
VTB_test_set = pd.read_csv("dataset/VTB_test_set.csv", delimiter=";")
test_set = np.concatenate((VTA_test_set,VTB_test_set))

#Create csv
np.savetxt('dataset/training_set.csv',training_set,fmt="%s",delimiter=";")
np.savetxt('dataset/test_set.csv',test_set,fmt="%s",delimiter=";")
