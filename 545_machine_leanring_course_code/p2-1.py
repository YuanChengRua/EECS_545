import numpy as np
import matplotlib.pyplot as plt


def plot_contour(Z, xlim, ylim, xlabel=None, ylabel=None):
  '''
    Z: function values of dimension Nx by Ny 
       where Nx and Ny are number of grid bins for x axis and y axis respectively 
    xlim: x-axis limit range
    ylim: y-axis limit range 
  '''
  cs = plt.contour(Z, extent=xlim+ylim)
  plt.gca().set_aspect("equal")
  plt.grid()
  if xlabel is not None:
    plt.xlabel(xlabel)  
  if ylabel is not None:
    plt.ylabel(ylabel)
    plt.clabel(cs, inline=0.1, fontsize=10)

  plt.show()
  return 


def multivariate_gaussian(x, mu, sigma):
  """
  Multivariate gaussian X ~ N(mu, sigma)
  
  Return:
  p(x): the Gaussian density at point x
  """
  part1 = np.exp(-1/2 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu)))
  part2 = ((2 * np.pi) ** (x.shape[0]/2) * (np.linalg.det(sigma) ** 0.5))


  #TODO (part 2a): Compute gaussian density
  pro_density =  part1 / part2

  return pro_density

def conditional_distribution(u_idx, u, mu, sigma): 
  """
  Arguments:
    u_idx: indices of the variables to condition on u: values of those variables to condition on mu: mean of the joint Gaussian
    sigma: covariance of the joint Gaussian
  Returns:
    cond_mu: conditional mean of p(v|u) cond_sigma: conditional covariance of p(v|u)
  """
  #TODO (part 2b): Compute conditional distribution
  p = sigma.shape[0]

  p_u = u_idx.shape[0]


  temp_idx_u = u_idx.tolist()
  temp_idx_v = []
  mu = mu.T

  mu_u = mu[temp_idx_u]
  for idx in range(p):
    if idx not in temp_idx_u:
      temp_idx_v.append(idx)

  mu_v = mu[temp_idx_v]

  new_orders = [1,2,0,3]
  new_sigma = np.zeros((4, 4))
  for i in range(4):
    for j in range(4):
      new_sigma[i][j] = sigma[new_orders[i]][new_orders[j]]

  sigma_v = new_sigma[p_u:, p_u:]
  # print(sigma_v)
  sigma_u = new_sigma[0:p_u, 0:p_u]
  # print(sigma_u)
  sigma_uv = new_sigma[0:p_u, p_u:]
  sigma_vu = new_sigma[p_u:, 0:p_u]
  # print(sigma_vu.shape)
  # print(sigma_uv.shape)
  # print(sigma_u.shape)

  temp1 = np.dot(sigma_vu, np.linalg.inv(sigma_u))
  print(temp1.shape)
  print(u.shape)
  temp2 = np.dot(temp1, (u - mu_u))

  cond_mean = mu_v + temp2
  cond_sigma = sigma_v - np.dot(np.dot(sigma_vu, np.linalg.inv(sigma_u)), sigma_uv)

  return cond_mean, cond_sigma

def prob_2c():
  
  # Data
  test_sigma_2 = np.array( [[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 1.5], [0.0, 0.0, 2.0, 0.0], [0.0, 1.5, 0.0, 4.0]])
  test_mu_2 = np.array([0.5, 0.0, -0.5, 0.0] )
  indices_2 = np.array([1,2])
  values_2 = np.array([0.1,-0.2])
  cond_mean, cond_sigma = conditional_distribution(indices_2, values_2, test_mu_2, test_sigma_2)
  cond_mean = cond_mean.reshape(-1,1)
  print(cond_sigma)
  print(cond_mean)
  #TODO (part 2e): plot the contour plot of the conditionla distirbution function with the provided grid space
  xlim = [-3.0, 3.0]
  ylim = [-3.0, 3.0]
  xgrid = np.linspace(*xlim, 100)
  ygrid = np.linspace(*ylim, 100)

  Z = []
  for idx in range(xgrid.shape[0]):
    for i in range(ygrid.shape[0]):
      temp = np.vstack([xgrid[idx], ygrid[i]])

      Z.append(multivariate_gaussian(temp, cond_mean, cond_sigma))


  Z = np.vstack(Z)
  Z = Z.reshape(xgrid.shape[0], ygrid.shape[0])
  plot_contour(Z, xlim, ylim)

if __name__ == '__main__':
  prob_2c()