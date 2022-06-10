import numpy as np 
from numpy.random import multivariate_normal as mn 
import matplotlib.pyplot as plt

from p2 import conditional_distribution

def gaussian_kernel(xx, sigma): 
  #TODO (part 3a): Compute kernel given a vector of inputs

  x1 = xx
  x2 = xx
  temp_residue = np.subtract.outer(x1, x2)
  SE_k = np.exp(-0.5 * (1/sigma ** 2) * temp_residue ** 2)

  return SE_k

def prob_3a(seed):
  np.random.seed(seed)
  N=100
  num_realizations = 3
  xx = np.linspace(-5, 5,N)
  sigma_list = [0.3, 0.5, 1.0]

  means = []
  for i in range(100):
    means.append(0)

  #TODO (part 3a):
  for sigma in sigma_list:
    plt.figure(figsize=(6, 4))
    cov_mat = gaussian_kernel(xx, sigma)
    for i in range(num_realizations):
      Y = np.random.multivariate_normal(means, cov_mat)

      plt.plot(xx, Y, linestyle='-', marker='o', markersize=3)
    plt.show()

  return 0

def prob_3c(seed):
  np.random.seed(seed)
  N=100
  D = np.array([[-1.3, 2],[ 2.4, 5.2] , [-2.5, -1.5] , [-3.3, -0.8] , [ 0.3, 0.3]])
  xx = np.linspace(-5, 5, N)
  sigma_list = [0.3, 0.5, 1.0]
  xx_jt = np.hstack((D[:,0], xx))  # Combine train
  mean_jt = np.zeros(D.shape[0] + xx.shape[0])
  num_realizations = 5

  obs_idx = np.array([0,1,2,3,4])
  obs_values = np.array([2, 5.2, -1.5, -0.8, 0.3])

  for idx, sigma in enumerate(sigma_list):
    K = gaussian_kernel(xx_jt, sigma)
    cond_mean, cond_sigma = conditional_distribution(obs_idx, obs_values, mean_jt, K)
    plt.figure(figsize=(6, 4))
    plt.plot(xx, cond_mean, linewidth = 4)
    for i in range(num_realizations):
      y = np.random.multivariate_normal(cond_mean.reshape(100, ), cond_sigma)
      plt.plot(xx, y)

    plt.plot(D[:, 0], D[:, 1], 'o', markersize=8, label='Data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
  prob_3a(seed=0)
  prob_3c(seed=0)

