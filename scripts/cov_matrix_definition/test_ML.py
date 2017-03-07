import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys

# Parameters
a = 1.0                                                 # angular coefficient
b = 2.5                                                 # linear coefficient
sig = 100

# Data
n_D = 50                                                 # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])


n           = []                                        # number of simulations
sigma_ML    = []
sigma_m1_ML = []

for n_S in range(n_D+3, n_D+50, 1):

    n.append(n_S)                                             # number of data points

    y2 = multivariate_normal.rvs(mean=yreal, cov=cov, size=n_S)
    # y2[:,j] = realisations for j-th data entry
    # y2[i,:] = data vector for i-th realisation

    # Estimate mean (ML)
    ymean = np.mean(y2, axis=0)

    # calculate covariance matrix
    cov_est = np.cov(y2,rowvar=False)

    # Double check that it's the same as calculating 'by hand'
    cov_est2 = np.zeros(shape=(n_D, n_D))
    if n_D > 1:
        for i in range(n_D):
            for j in range(n_D):
                cov_est2[i,j] = sum((y2[:,i] - ymean[i]) * (y2[:,j] - ymean[j])) / (n_S - 1.0)
    else:
        cov_est2[0,0] = sum((y2 - ymean) * (y2 - ymean)) / (n_S - 1.0)
        cov_est = [[cov_est]]

    # Normalised trace
    this_sigma_ML = np.trace(cov_est) / n_D
    sigma_ML.append(this_sigma_ML)

    cov_est_inv = np.linalg.inv(cov_est)

    this_sigma_m1_ML = np.trace(cov_est_inv) / n_D
    sigma_m1_ML.append(this_sigma_m1_ML)

    print n_S, this_sigma_ML, this_sigma_m1_ML


plt.figure()
plt.suptitle('Covariance of n_D = {} dimensional data'.format(n_D))

plt.subplot(1, 2, 1)
plt.plot(n, sigma_ML, 'b.')
plt.plot([n[0], n[-1]], [sig, sig], 'r-')
plt.xlabel('n_S')
plt.ylabel('normalised trace of ML covariance')

plt.subplot(1, 2, 2)
plt.plot(n, sigma_m1_ML, 'b.')
plt.plot([n[0], n[-1]], [1.0/sig, 1.0/sig], 'r-')
bias = [(n_S-1.0)/(n_S-n_D-2.0)/sig for n_S in n]
plt.plot(n, bias, 'g-.')
plt.xlabel('n_S')
plt.ylabel('normalised trace of inverse of ML covariance')

plt.savefig('sigma_ML')


