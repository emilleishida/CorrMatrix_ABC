# Adapted from: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
# (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
# 
# you are kindly asked to include the complete citation if you used this 
# material in a publication

# Code 4.3 - Normal linear model in Python using STAN
# 1 response (y) and 1 explanatory variable (x1)

import numpy as np
import statsmodels.api as sm
import pystan
from scipy.stats import uniform, multivariate_normal
import time
import pylab as plt

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 100                          # number of obs in model 
nsim = 1000
x1 = uniform.rvs(loc=-100, scale=200, size=nobs)          # random uniform variable

x1.transpose()                   # create response matrix
X = sm.add_constant(x1)          # add intercept
beta = [2.5, 1.0]                # create vector of parameters

xb = np.dot(X, beta)                                  # linear predictor, xb
y = np.random.normal(loc=xb, scale=1.0, size=nobs)    # create y as adjusted
      
cov = np.diag([1 for i in range(nobs)]) 
yreal = beta[1] * x1 + beta[0]

# generate many simulations
y2 = []
for k in range(nsim):
    print 'k = ' + str(k)

    y2.append(multivariate_normal.rvs(mean=yreal, cov=cov))

y2 = np.array(y2)
y3 = []
for i in range(y2.shape[1]):
    line = []
    for j in range(y2.shape[0]):
        line.append(y2[j][i])
    y3.append(line)

y3 = np.array(y3)    

# estimate covariance matrix
cov = np.cov(y3)


diag = np.diag(cov)

nondiag = []
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        if i != j:
            nondiag.append(cov[i][j])

offdiag = nondiag

dmean = np.mean(diag)
dstd = np.std(diag)

odmean = np.mean(offdiag)
odstd = np.std(offdiag)


# plot
fig = plt.figure(figsize=(20,15))
plt.suptitle('number of observations: ' + str(nobs) + ';      number of simulations:  ' + str(nsim), fontsize=22)

plt.subplot(1,2,1)
h1 = plt.hist(diag)
plt.text(0.92*max(diag), 1.01*max(h1[0]), '$\sigma_{ii}^2$ = mean = ' + str(round(dmean, 4)), fontsize=22)
plt.text(0.95*max(diag), 0.97*max(h1[0]), 'std  = ' + str(round(dstd, 4)), fontsize=22)
plt.xticks(fontsize=22)
plt.yticks([])
plt.xlabel('diagonal', fontsize=26)
plt.ylabel('number of coefficients', fontsize=26)
plt.xlim(0.85,1.175)

plt.subplot(1,2,2)
h2 = plt.hist(offdiag)
plt.text(-0.25*min(offdiag), 1.01*max(h2[0]), '$\sigma_{ij}^2$ = mean = ' + str(round(odmean, 4)), fontsize=22)
plt.text(-0.55*min(offdiag), 0.97*max(h2[0]), 'std  = ' + str(round(odstd, 4)), fontsize=22)
plt.xticks(fontsize=22)
plt.yticks([])
plt.xlabel('off diagonal', fontsize=26)
plt.ylabel('number of coefficients', fontsize=26)
plt.xlim(-0.175, 0.175)

fig.subplots_adjust(left=0.07, right=0.94, bottom=0.075, top=0.9, hspace=0.15, wspace=0.25)
plt.savefig('covariance_matrix.png')

# Fit
toy_data = {}                  # build data dictionary
toy_data['nobs'] = nobs        # sample size
toy_data['x'] = x1             # explanatory variable
toy_data['y'] = y2[0]          # response variable
toy_data['cov'] = cov

# STAN code
stan_code = """
data {
    int<lower=0> nobs;                                 
    vector[nobs] x;                       
    vector[nobs] y;                       
    cov_matrix[nobs] cov;
}
parameters {
    real a;
    real b;                                                
    real<lower=0> sigma;               
}
model {
    vector[nobs] mu;

    mu = b + a * x;

    y ~ multi_normal(mu, sigma*cov);             # Likelihood function
}
"""
start = time.time()
fit = pystan.stan(model_code=stan_code, data=toy_data, iter=5000, chains=3, verbose=False, n_jobs=3)
end = time.time()

elapsed = end - start

print 'elapsed time = ' + str(elapsed)

# Output
nlines = 8                     # number of lines in screen output

output = str(fit).split('\n')
for item in output[:nlines]:
    print(item)   


# Plot
import pylab as plt

fit.plot(['a', 'b', 'sigma'])
plt.tight_layout()
plt.savefig('posteriors_MCMC_correlated.png')
