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
from scipy.stats import uniform
import time

# Data
np.random.seed(1056)                 # set seed to replicate example
nobs= 3000                           # number of obs in model 
x1 = uniform.rvs(loc=-100, scale=200, size=nobs)          # random uniform variable

x1.transpose()                   # create response matrix
X = sm.add_constant(x1)          # add intercept
beta = [2.5, 1.0]                # create vector of parameters

xb = np.dot(X, beta)                                  # linear predictor, xb
y = np.random.normal(loc=xb, scale=1.0, size=nobs)    # create y as adjusted
                                                      # random normal variate 

# Fit
toy_data = {}                  # build data dictionary
toy_data['nobs'] = nobs        # sample size
toy_data['x'] = x1             # explanatory variable
toy_data['y'] = y              # response variable

# STAN code
stan_code = """
data {
    int<lower=0> nobs;                                 
    vector[nobs] x;                       
    vector[nobs] y;                       
}
parameters {
    real a;
    real b;                                                
    real<lower=0> sigma;               
}
model {
    vector[nobs] mu;

    mu = b + a * x;

    y ~ normal(mu, sigma);             # Likelihood function
}
"""
start = time.time()
fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2500, chains=3, verbose=False, n_jobs=3)
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
plt.savefig('posteriors_MCMC.png')
