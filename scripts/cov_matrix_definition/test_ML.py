import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys
import pystan
import time



def get_cov_ML_by_hand(mean, cov, size):
    """ Double check that get_cov_ML is the same as calculating 'by hand'"""

    cov_est = np.zeros(shape=(size, size))
    if size > 1:
        for i in range(size):
            for j in range(size):
                cov_est[i,j] = sum((y2[:,i] - ymean[i]) * (y2[:,j] - ymean[j])) / (size - 1.0)
    else:
        cov_est[0,0] = sum((y2 - ymean) * (y2 - ymean)) / (size - 1.0)

    return cov_est



def get_cov_ML(mean, cov, size):
    """
    *** covariance of the data! ***
    Simulate a set of catalogs from  a multivariate gaussian with given mean,
    covariance and number of observations. 
    Then extract the mean and covariance from the simulated values.

    input: mean, expected mean of observations - array of floats
           cov, covariance matrix  - matrix of floats
           size, number of observations in the catalog - int

    output: cov_est, the estimated covariance matrix - matrix of floats
    """

    # simulate catalog
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



def plot_sigma_ML(n, sigma_ML, sigma_m1_ML, sig):

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



def fit(x1, cov):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit  without taking the correlation into
    consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov, square covariance matrix for the multivariate normal

    output: fit, Stan fitting object
    """

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable

    # cov = covariance of the data!
    y = multivariate_normal.rvs(mean=x1, cov=cov, size=1)
    toy_data['y'] = y                        # response variable, here one realisation
    toy_data['sigma'] = np.sqrt(cov[0][0])   # scatter is not a parameter to be estimated

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs; 
        real<lower=0> sigma;                                
        vector[nobs] x;                       
        vector[nobs] y;                       
    }
    parameters {
        real a;
        real b;                                                              
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

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit

def fit_corr(x1, cov_true, cov_estimated):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_estimated, square covariance matrix for the fitting

    output: fit, Stan fitting object
    """

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable

    # cov = covariance of the data!
    y = multivariate_normal.rvs(mean=x1, cov=cov_true, size=1)
    toy_data['y'] = y              # response variable, here one realisation

    # set estimated covariance matrix for fitting
    toy_data['cov_est'] = cov_estimated

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs;                                 
        vector[nobs] x;                       
        vector[nobs] y;   
        matrix[nobs, nobs] cov_est;                    
    }
    parameters {
        real a;
        real b;                                                              
    }
    model {
        vector[nobs] mu;

        mu = b + a * x;

        y ~ multi_normal(mu, cov_est);             # Likelihood function
    }
    """

    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2500, chains=3, verbose=False, n_jobs=3)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit



# Parameters
a = 1.0                                                 # angular coefficient
b = 0                                                 # linear coefficient
sig = 1                                               # *** cov of the data! ***

# Data
n_D = 50                                                 # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])            # *** cov of the data in the same catalog! ***


n           = []                                        # number of simulations
sigma_ML    = []
sigma_m1_ML = []
fit_res     = {}
fit_res['a_mean'] = []
fit_res['a_std'] = []
fit_res['b_mean'] = []

for n_S in range(n_D+3, n_D+200, 10):

    n.append(n_S)                                             # number of data points

    cov_est = get_cov_ML(yreal, cov, n_S)

    # Normalised trace
    this_sigma_ML = np.trace(cov_est) / n_D
    sigma_ML.append(this_sigma_ML)

    cov_est_inv = np.linalg.inv(cov_est)

    this_sigma_m1_ML = np.trace(cov_est_inv) / n_D
    sigma_m1_ML.append(this_sigma_m1_ML)

    #print n_S, n_D, len(x1), this_sigma_ML, this_sigma_m1_ML

    # MCMC fit of parameters
    res = fit_corr(x1, cov, cov_est)
    la  = res.extract(permuted=True)
    fit_res['a_mean'].append(np.mean(la['a']))
    fit_res['a_std'].append(np.std(la['a']))
    fit_res['b_mean'].append(np.mean(la['b']))


plot_sigma_ML(n, sigma_ML, sigma_m1_ML, sig)

print(fit_res['a_mean'])
print(fit_res['a_std'])
print(fit_res['b_mean'])

plt.figure()
plt.subplot(1,2,1)
plt.scatter(n, fit_res['a_mean'], color='b')
plt.plot([n[0], n[-1]], [a, a], color='r', label='true value')
plt.xlabel('n_S')
plt.ylabel('a_mean')
plt.ylim(min(fit_res['b_mean']) - 1, max(fit_res['b_mean'])+1)

plt.subplot(1,2,2)
plt.scatter(n, fit_res['b_mean'], color='b')
plt.plot([n[0], n[-1]], [b, b], color='r', label='true value')
plt.xlabel('n_S')
plt.ylabel('b_mean')
plt.ylim(min(fit_res['b_mean']) - 1, max(fit_res['b_mean'])+1)
plt.savefig('output_from_Stan.png')


