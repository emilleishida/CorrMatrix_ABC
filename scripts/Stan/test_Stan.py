import pystan
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

def get_cov_ML(mean, cov, size):

    y2 = multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    # y2[:,j] = realisations for j-th data entry
    # y2[i,:] = data vector for i-th realisation

    # Estimate mean (ML)
    ymean = np.mean(y2, axis=0)

    # calculate covariance matrix
    cov_est = np.cov(y2, rowvar=False)

    if size > 1:
        pass
    else:
        cov_est = [[cov_est]]

    return cov_est

# Parameters
a = 1.0                                                 # angular coefficient
b = 0                                                 # linear coefficient
sig = 10

# Data
n_D = 500        
n_S = 1500                                        # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])            # *** cov of the data in the same catalog! ***

cov_est = get_cov_ML(yreal, cov, n_S)


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

fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2500, chains=3, verbose=False, n_jobs=3)


print(fit)
