import pystan
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

# Parameters
a = 1.0                                                 # angular coefficient
b = 0                                                 # linear coefficient
sig = 5

# Data
n_D = 750                                              # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])            # *** cov of the data in the same catalog! ***

# Fit
toy_data = {}                  # build data dictionary
toy_data['nobs'] = len(x1)     # sample size = n_D
toy_data['x'] = x1             # explanatory variable

y = multivariate_normal.rvs(mean=x1, cov=cov, size=1)
toy_data['y'] = y              # response variable, here one realisation

# set estimated covariance matrix for fitting
toy_data['cov'] = cov

# STAN code
# the fitting code does not believe that observations are correlated!
stan_code = """
data {
    int<lower=0> nobs;                                 
    vector[nobs] x;                       
    vector[nobs] y;   
    matrix[nobs, nobs] cov;                    
}
parameters {
    real a;
    real b;                                                              
}
model {
    vector[nobs] mu;
    mu = b + a * x;

    y ~ multi_normal(mu, cov);             # Likelihood function
}
"""

fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=3, verbose=False, n_jobs=3)

print(fit)


# this returns the result bellow, so we see that when we use the same covariance matrix 
# for simulation and for the fitting with 750 data points and sig=5 the results
# from Stan are reasonable... 
# but we cannot expect the scatter of b to be much smaller than 0.1

"""
Inference for Stan model: anon_model_e9f10baf834c1214e3f5640ab86ee950.
3 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=3000.

       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
a       1.0  2.8e-5 1.4e-3    1.0    1.0    1.0    1.0    1.0 2589.0    1.0
b   -9.8e-3  2.5e-3   0.08  -0.17  -0.07-9.6e-3   0.05   0.14 1003.0   1.01
lp__ -381.8    0.02   0.97 -384.3 -382.2 -381.5 -381.1 -380.8 1634.0    1.0

Samples were drawn using NUTS at Fri Apr 21 18:53:36 2017.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
"""

