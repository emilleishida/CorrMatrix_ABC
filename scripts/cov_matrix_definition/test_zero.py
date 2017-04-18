import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt

# data
a = 1.0                                                 # angular coefficient
b = 2.5                                                 # linear coefficient
nobs = 1000                                             # number of data points

x1 = uniform.rvs(loc=-100, scale=200, size=nobs)        # exploratory variable
x1.sort()

y1_err = norm.rvs(loc=0, scale=10, size=nobs)           # noise

yreal = a * x1 + b
y1 = yreal + y1_err
cov = np.diag([100 for i in range(nobs)])
y2 = multivariate_normal.rvs(mean=yreal, cov=cov)

# calculate covariance matrix
cov_est = []
for i in range(nobs):
    line = []
    for j in range(nobs):
        line.append((y2[i] - yreal[i]) * (y2[j] - yreal[j]))
    cov_est.append(line)

cov_est = np.array(cov_est)

# plot
plt.figure()
plt.suptitle('Model 1: independent realizations')
plt.subplot(1,2,1)
plt.scatter(x1, y1, s=2.0, label='data', alpha=0.25)
plt.plot(x1, yreal,'--', color='green', label = 'model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', frameon=False)

plt.subplot(1,2,2)
plt.hist(y1_err)
plt.xlabel('$\epsilon$')
plt.savefig('test_normal.png')


plt.figure()
plt.suptitle('Model 2: considering covariance matrix')
plt.subplot(1,2,1)
plt.scatter(x1, y2, s=2.0, label='data', alpha=0.25)
plt.plot(x1, yreal,'--', color='green', label = 'model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', frameon=False)

plt.subplot(1,2,2)
plt.hist(y2-yreal)
plt.xlabel('measured - model')
plt.savefig('test_multi_normal.png')
