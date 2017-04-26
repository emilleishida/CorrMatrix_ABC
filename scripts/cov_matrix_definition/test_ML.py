import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys
import pystan
import time

# Gaussian likelihood function
# chi^2 = -2 log L = (y - mu)^t Psi (y - mu).
# y: n_D dimensional data vector, simulated as y(x) ~ N(mu(x), sigma)
#    with mu(x) = b + a * x
# x: x ~ Uniform(-100, 100)
# mu: mean, mu(x) = b + a * x, with parameters b, a
# Psi: estimated inverse covariance, Phi^-1 times correction factor
# Phi: estimate of true covariance C, ML estimate from n_S realisations of y.
# C = diag(sig, ..., sig)
# Fisher matrix
# F_rs = 1/2 ( dmu^t/dr Psi dmu/ds + dmu^t/ds Psi dmu/dr)
#      = 1/2 |2 x^t Psi x               x^t Psi 1 + 1^t Psi x| 
#            |1^t Psi x + x^t Psi 1     2 1^t Psi 1|
# e.g. F_11 = F_aa = x Psi x^t

def Fisher_ana_ele(r, s, y, Psi):
    """Return analytical Fisher matrix element (r, s).

    Parameters
    ----------
    r, s: integer
        indices of matrix, r,s in {0,1}
    y: array of float
        data vector
    Psi: matrix
        precision matrix

    Returns
    -------
    f_rs: float
        Fisher matrix element (r, s)
    """

    n_D = len(y)
    v = np.zeros(shape = (2, n_D))
    for i in (r, s):
        if i == 0:
            v[i] = y
        elif i == 1:
            v[i] = np.ones(shape=n_D)
        else:
            print('Invalid index {}'.format(i))
            sys.exit(1)

    f_rs = np.einsum('i,ij,j', v[r], Psi, v[s])
    # Check result by hand
    #f_rs = 0
    #for i in range(n_D):
        #for j in range(n_D):
            #f_rs += v[r,i] * Psi[i, j] * v[s, j]

    return f_rs


def Fisher_error(F):
    """Return errors (Cramer-Rao bounds) from Fisher matrix

    Parameters
    ----------
    F: matrix of float
        Fisher matrix

    Returns
    -------
    d: array of float
        vector of parameter errors
    """

    Finv = np.linalg.inv(F)

    return np.sqrt(np.diag(Finv))



def Fisher_ana(y, Psi):
    """Return analytical Fisher matrix

    Parameters
    ----------
     y: array of float
        data vector
    Psi: matrix
        precision matrix

    Returns
    -------
    f: matrix
        Fisher matrix
    """

    f = np.zeros(shape = (2, 2))
    for r in (0, 1):
        for s in (0, 1):
            f[r,s] = Fisher_ana_ele(r, s, y, Psi)

    return f



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

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable
    y = multivariate_normal.rvs(mean=x1, cov=cov, size=1)
    toy_data['y'] = y              # response variable, here one realisation


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

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit


# Main program

# Parameters
a = 1.0                                                 # angular coefficient
b = 0                                                 # linear coefficient
sig = 5
do_fit_stan = True

#np.random.seed(1056)                 # set seed to replicate example


# Data
n_D = 750                                                 # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])            # *** cov of the data in the same catalog! ***


n           = []                                        # number of simulations
sigma_ML    = []
sigma_m1_ML = []
fit_res     = {}

for var in ['a', 'b']:
    for t in ['mean', 'std']:
        fit_res['{}_{}'.format(var, t)] = []


for n_S in range(n_D+3, n_D+2750, 50):

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
    res = fit(x1, cov)
    la  = res.extract(permuted=True)
    fit_res['a_mean'].append(np.mean(la['a']))
    fit_res['a_std'].append(np.std(la['a']))


plot_sigma_ML(n, sigma_ML, sigma_m1_ML, sig)

print(fit_res['a_mean'])
print(fit_res['a_std'])


