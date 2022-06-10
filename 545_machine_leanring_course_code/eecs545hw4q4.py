# -*- coding: utf-8 -*-
"""EECS545HW4Q4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YqJN-lSautah3tXkuWDJDfrmeW0KQgC1
"""

import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt

# You have have to install the libraries below.
# sklearn, csv
import csv

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# The csv file air-quality-train.csv contains the training data.
# After loaded, each row of X_train will correspond to CO, NO2, O3, SO2.
# The vector y_train will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_train = []
y_train = []

with open('air-quality-train.csv', 'r') as air_quality_train:
    air_quality_train_reader = csv.reader(air_quality_train)
    next(air_quality_train_reader)
    for row in air_quality_train_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_train.append([row[1], row[2], row[3], row[4]])
        y_train.append(row[5])
        
# The csv file air-quality-test.csv contains the testing data.
# After loaded, each row of X_test will correspond to CO, NO2, O3, SO2.
# The vector y_test will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_test = []
y_test = []

with open('air-quality-test.csv', 'r') as air_quality_test:
    air_quality_test_reader = csv.reader(air_quality_test)
    next(air_quality_test_reader)
    for row in air_quality_test_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_test.append([row[1], row[2], row[3], row[4]])
        y_test.append(row[5])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# TODOs for part (a)
#    1. Use SVR loaded to train a SVR model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset
svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
y_hat = svr_rbf.fit(X_train, y_train).predict(X_test)

mse_a = mean_squared_error(y_test, y_hat)
print("The rmse of SVR with C = 1, gamma = 0.1 is {}".format(np.sqrt(mse_a)))

# TODOs for part (b)
#    1. Use KernelRidge to train a Kernel Ridge  model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset 
kerridge = KernelRidge(kernel='rbf', alpha=0.5, gamma=0.1)
y_hat_b = kerridge.fit(X_train, y_train).predict(X_test)
mse_b = mean_squared_error(y_test, y_hat_b)


print("The rmse of KR with alpha = 0.5, gamma = 0.1 is {}".format(np.sqrt(mse_b)))

y_train = y_train.reshape(-1, 1)

# Use this seed.
seed = 0
np.random.seed(seed) 
trainset = np.hstack((X_train, y_train))
np.random.shuffle(trainset)

K = 5 #The number of folds we will create 

# TODOs for part (c)
#   1. Create a partition of training data into K=5 folds 
#   Hint: it suffice to create 5 subarrays of indices   
fold_size = int(X_train.shape[0] / K)

partitions = []
for i in range(K):
  partitions.append(trainset[i * fold_size : (i + 1) * fold_size])


reg_range = np.logspace(-1,1,3)     # Regularization paramters
kpara_range = np.logspace(-2, 0, 3) # Kernel parameters 

# # TODOs for part (d)
# #    1.  Select the best parameters for both SVR and KernelRidge based on k-fold cross-validation error estimate (use RMSE as the performance metric)
# #    2.  Print the best paramters for both SVR and KernelRidge selected  
# #    3.  Train both SVR and KernelRidge on the full training data with selected best parameters 
# #    4.  Print both the RMSE on the test dataset of SVR and KernelRidge 

est_error_svr = []
est_error_kr = []
for reg in reg_range:
  for kpara in kpara_range:
    temp_est_error_svr = []
    temp_est_error_kr = []
    for idx, ele in enumerate(partitions):
      validation = ele
      val_x = validation[:,0:4]
      val_y = validation[:,-1]

      temp = []
      for ele_temp in partitions:
        if ele_temp.all == ele.all:
          continue
        else:
          temp.append(ele_temp)

      train = np.vstack(temp)
      train_x = train[:,0:4]
      train_y = train[:, -1]

      svr_temp = SVR(kernel='rbf', C=reg, gamma=kpara)
      svr_temp.fit(train_x, train_y)
      y_hat_temp_svr = svr_temp.predict(val_x)
      temp_est_error_svr.append(mean_squared_error(val_y,y_hat_temp_svr))

      kr_temp = KernelRidge(kernel='rbf', alpha=1/(2*reg), gamma=kpara)
      kr_temp.fit(train_x, train_y)
      y_hat_temp_kr = kr_temp.predict(val_x)
      temp_est_error_kr.append(mean_squared_error(val_y, y_hat_temp_kr))



    est_error_svr.append((reg, kpara, sum(temp_est_error_svr) / len(temp_est_error_svr)))
    est_error_kr.append((1/(2*reg), kpara, sum(temp_est_error_kr) / len(temp_est_error_kr)))

# est_error_kr
# # est_error_svr

print(est_error_kr)
print(est_error_svr)

# We can see the that for KR, the optimal parameters are: alpha = 0.05, gamma = 0.01
# The optimal parameters for SVR are: C = 10, gamma = 0.01

svr_rbf_d = SVR(kernel='rbf', C=10, gamma=0.01)
y_hat_d = svr_rbf_d.fit(X_train, y_train).predict(X_test)

mse_d = mean_squared_error(y_test, y_hat_d)
rmse_svr = np.sqrt(mse_d)
print("The rmse of SVR with C = 10, gamma = 0.01 is {}".format(rmse_svr))

kerridge_d = KernelRidge(kernel='rbf', alpha=0.05, gamma=0.01)
y_hat_kr_d = kerridge_d.fit(X_train, y_train).predict(X_test)
mse_kr_d = mean_squared_error(y_test, y_hat_kr_d)
rmse_kr = np.sqrt(mse_kr_d)

print("The rmse of KR with alpha = 0.05, gamma = 0.01 is {}".format(rmse_kr))
