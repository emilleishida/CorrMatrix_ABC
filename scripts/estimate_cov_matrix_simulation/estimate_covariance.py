import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt

# data
a = 1.0                                                 # angular coefficient
b = 2.5                                                 # linear coefficient
nobs = 1000                                             # number of data points
nsim = 100                                            # number of simulations

x1 = uniform.rvs(loc=-100, scale=200, size=nobs)        # exploratory variable
x1.sort()

y1_err = norm.rvs(loc=0, scale=1, size=nobs)           # noise

yreal = a * x1 + b
y1 = yreal + y1_err
cov = np.diag([1 for i in range(nobs)])

# generate many simulations
y2 = []
for k in range(nsim):
    print 'k = ' + str(k)

    y2.append(multivariate_normal.rvs(mean=yreal, cov=cov))




cmean = []
for i in range(nobs):
    line = []
    for j in range(nobs):
        line.append(np.mean([cov_estimate[k][i][j] for k in range(nsim)]))
    cmean.append(line)

cmean = np.array(cmean)


diag = np.diag(cmean)
offdiag = cmean - np.diag(diag)
offdiag = offdiag.flatten()

dmean = np.mean(diag)
dstd = np.std(diag)

odmean = np.mean(offdiag)
odstd = np.std(offdiag)

# plot
fig = plt.figure(figsize=(20,15))
plt.suptitle('number of observations: ' + str(nobs) + ';      number of simulations:  ' + str(nsim), fontsize=22)

plt.subplot(1,2,1)
h1 = plt.hist(diag)
plt.text(2.05*min(diag), 1.01*max(h1[0]), '$\sigma_{ii}^2$ = mean = ' + str(round(dmean, 4)), fontsize=22)
plt.text(2.3*min(diag), 0.975*max(h1[0]), 'std  = ' + str(round(dstd, 4)), fontsize=22)
plt.xticks(fontsize=22)
plt.yticks([])
plt.xlabel('diagonal', fontsize=26)
plt.ylabel('number of simulations', fontsize=26)

plt.subplot(1,2,2)
h2 = plt.hist(offdiag)
plt.text(-0.25*min(offdiag), 1.01*max(h2[0]), '$\sigma_{ij}^2$ = mean = ' + str(round(odmean, 4)), fontsize=22)
plt.text(-0.55*min(offdiag), 0.97*max(h2[0]), 'std  = ' + str(round(odstd, 4)), fontsize=22)
plt.xticks(fontsize=22)
plt.yticks([])
plt.xlabel('off diagonal', fontsize=26)
plt.ylabel('number of simulations', fontsize=26)

fig.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.9, hspace=0.15, wspace=0.25)
plt.savefig('covariance_matrix.png')



