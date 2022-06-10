# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

train_x = np.load('E:\FALL_dataset\hw2p2_data\hw2p2_train_x.npy')
train_y = np.load('E:\FALL_dataset\hw2p2_data\hw2p2_train_y.npy')

n, d = train_x.shape

train_y = train_y.reshape((1192,1))
complete_train = np.concatenate((train_x, train_y), axis = 1)

class_1_list = []
class_0_list = []
for i in range(complete_train.shape[0]):
  if complete_train[i, -1] == 1:
    class_1_list.append(complete_train[i,:])
  else:
    class_0_list.append(complete_train[i,:])
class_1 = np.vstack(class_1_list)
class_0 = np.vstack(class_0_list)

n1 = np.sum(class_1[:, 0:-1])
n0 = np.sum(class_0[:,0:-1])

n1j = np.sum(class_1, axis = 0)
n1j = n1j[0:-1]

n0j = np.sum(class_0, axis = 0)
n0j = n0j[0:-1]
# n0j = n0j.reshape(1,1000)
# n1j = n1j.reshape(1,1000)

p1j_hat = (n1j + 1) / (n1 + 1 * 1000)

p0j_hat = (n0j + 1) / (n0 + 1 * 1000)

pi_0 = class_0.shape[0] / n
pi_1 = class_1.shape[0] / n

# In the question 2, I calculate the estimation of pkj for each class and for each feature and also the log prior for 
# each class

log_p0j_hat = np.log(p0j_hat)
log_p0j_hat = log_p0j_hat.reshape(1000, 1)

log_p1j_hat = np.log(p1j_hat)
log_p1j_hat = log_p1j_hat.reshape(1000, 1)

log_pi_0 = np.log(pi_0)
log_pi_1 = np.log(pi_1)

test_x = np.load('E:\FALL_dataset\hw2p2_data\hw2p2_test_x.npy')
test_y = np.load('E:\FALL_dataset\hw2p2_data\hw2p2_test_y.npy')


# Generate the results from the testing dataset 

generate_result = []

for i in range(test_x.shape[0]):
  small_pice = test_x[i,:]
  small_pice = small_pice.reshape(1, 1000)
  temp1 = np.dot(small_pice, log_p1j_hat) + log_pi_1
  temp0 = np.dot(small_pice, log_p0j_hat) + log_pi_0
  if temp1 > temp0:
    generate_result.append(1)
  else:
    generate_result.append(0)

# We can see that the testing error is 0.1259.
error_counter = 0
for i in range(len(generate_result)):
  if generate_result[i] != test_y[i]:
    error_counter = error_counter + 1

test_error_rate = error_counter / len(generate_result)
print(test_error_rate)

# e) if we guess all the outcomes of the testing dataset are 1s
error_counter_1 = 0

for i in range(test_y.shape[0]):
  if test_y[i] != 1:
    error_counter_1 = error_counter_1 + 1
error_rate_all_1 = error_counter_1 / test_y.shape[0]
print(error_rate_all_1)


