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



def Fisher_num_ele(r, s, x, a, b, Psi):
    """Return numerical Fisher matrix element (r, s).

    Parameters
    ----------
    r, s: integer
        indices of matrix, r,s in {0,1}
    x: array of float
        x-values on which data vector is evaluated
    a, b: double
        parameters
    Psi: matrix
        precision matrix

    Returns
    -------
    f_rs: float
        Fisher matrix element (r, s)
    """


    if r == 0:
        h = 0.1 * a
        dy_dr = (a+h * x + b - (a-h * x + b)) / (2*h)
    else:
        h = 0.1
        dy_dr = (a * x + b+h - (a * x + b-h)) / (2*h)

    if s == 0:
        h = 0.1 * a
        dy_ds = (a+h * x + b - (a-h * x + b)) / (2*h)
    else:
        h = 0.1
        dy_ds = (a * x + b+h - (a * x + b-h)) / (2*h)

    f_rs = np.einsum('i,ij,j', dy_dr, Psi, dy_ds)

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



def Fisher_num(x, a, b, Psi):
    """Return numerical Fisher matrix

    Parameters
    ----------
    x: array of float
        x-values of data vector
    a, b: double
        parameters
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
            f[r,s] = Fisher_num_ele(r, s, x, a, b, Psi)

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



def debias_cov(cov_est_inv, n_S):
    """Return debiased inverse covariance estimate
    """

    n_D   = cov_est_inv.shape[0]
    if n_S - n_D - 2 <= 0:
        print('Number of simulations {} too small for data dimension = {}, resulting in singular covariance'.format(n_S, n_D))
    alpha = (n_S - n_D - 2.0) / (n_S - 1.0)

    return alpha * cov_est_inv



def plot_sigma_ML(n, n_D, sigma_ML, sigma_m1_ML, sig, out_name='sigma_ML'):

    plt.figure()
    plt.suptitle('Covariance of n_D = {} dimensional data'.format(n_D))

    plt.subplot(1, 2, 1)
    plt.plot(n, sigma_ML, 'b.')
    plt.plot([n[0], n[-1]], [sig, sig], 'r-')
    plt.xlabel('n_S')
    plt.ylabel('normalised trace of ML covariance')

    ax = plt.subplot(1, 2, 2)
    plt.plot(n, sigma_m1_ML, 'b.')
    plt.plot([n[0], n[-1]], [1.0/sig, 1.0/sig], 'r-')
    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
    bias = [(n_S-1.0)/(n_S-n_D-2.0)/sig for n_S in n_fine]
    plt.plot(n_fine, bias, 'g-.')
    plt.xlabel('n_S')
    plt.ylabel('normalised trace of inverse of ML covariance')
    #plt.ylim(90, 110)
    ax.set_yscale('log')

    plt.savefig('{}.pdf'.format(out_name))

    f = open('{}.txt'.format(out_name), 'w')
    print >>f, '# sig={}, n_D={}'.format(sig, n_D)
    print >>f, '# n sigma 1/sigma'
    for i in range(len(n)):
        print >>f, '{} {} {}'.format(n[i], sigma_ML[i], sigma_m1_ML[i])
    f.close()


def plot_mean_std(n, n_D, fit_res, out_name='line_mean_std', a=1, b=2.5):
    """Plot mean and std from MCMC fits versus number of
       realisations n

    Parameters
    ----------
    n: array of integer
        number of realisations {n_S} for ML covariance
    n_D: integer
        dimension of data vector
    fit_res: pystan.stan return object
        contains fit results
    a: float
        input value for intercept, default=1
    b: floag
        input value for slope, default=2.5
    out_name: string
        output file name base, default='line_mean_std'

    Returns
    -------
    None
    """

    plt.figure()
    plt.suptitle('Fit of straight line with $n_{{\\rm D}}$ = {} data points'.format(n_D))

    if 'a_mean' in fit_res and len(fit_res['a_mean']) > 0:
        plt.subplot(1, 2, 1)
        plt.plot(n, fit_res['a_mean'], 'b.')
        plt.plot([n[0], n[-1]], [a, a], 'r-')
        plt.plot(n, fit_res['b_mean'], 'bD', markersize=0.3)
        plt.plot([n[0], n[-1]], [b, b], 'r-')
        plt.xlabel('n_S')
        plt.ylabel('mean of intercept, slope')

    ax= plt.subplot(1, 2, 2)
    plt.plot(n, fit_res['a_std'], 'b.')
    plt.plot(n, fit_res['b_std'], 'bD')
    plt.xlabel('n_S')
    plt.ylabel('std of intercept, slope')
    ax.set_yscale('log')

    plt.savefig('{}.pdf'.format(out_name))

    if 'a_mean' in fit_res and len(fit_res['a_mean']) > 0:
        f = open('{}_mean.txt'.format(out_name), 'w')
        print >>f, '# n a b'
        for i in range(len(n)):
            print >>f, '{} {} {}'.format(n[i], fit_res['a_mean'][i], 
                    fit_res['b_mean'][i])
        f.close()

    f = open('{}_std.txt'.format(out_name), 'w')
    print >>f, '# n a_std b_std'
    for i in range(len(n)):
        print >>f, '{} {} {}'.format(n[i], fit_res['a_std'][i], fit_res['b_std'][i])
    f.close()





def fit(x1, cov, n_jobs=3):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit  without taking the correlation into
    consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov, square covariance matrix for the multivariate normal
            n_jobs, number of parallel jobs

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
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2500, chains=3, verbose=False, n_jobs=n_jobs)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit

def fit_corr(x1, cov_true, cov_estimated, n_jobs=3):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_estimated, square covariance matrix for the fitting
            n_jobs, number of parallel jobs

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
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=3, verbose=False, n_jobs=n_jobs)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit


# Main program

# Parameters
a = 1.0                                                 # angular coefficient
b = 0                                                 # linear coefficient
sig = 5

#do_fit_stan = True
do_fit_stan = False
n_jobs = 1

#np.random.seed(1056)                 # set seed to replicate example


# Data
n_D = 750                                                 # Dimension of data vector
x1 = uniform.rvs(loc=-100, scale=200, size=n_D)        # exploratory variable
x1.sort()

yreal = a * x1 + b
cov = np.diag([sig for i in range(n_D)])            # *** cov of the data in the same catalog! ***


n            = []                                        # number of simulations
sigma_ML     = []
sigma_m1_ML  = []

class Results:
    """Store results of sampling
    """

    def __init__(self, par_name, n_n_S, n_R):
        """Set arrays for mean and std storing all n_S simulation cases
           with n_R runs each
        """

        self.mean = {}
        self.std  = {}
        for p in par_name:
            self.mean[p] = np.zeros(shape = (n_n_S, n_R))
            self.std[p]  = np.zeros(shape = (n_n_S, n_R))


    def set_mean(self, par, par_name, i, run):
        """Set mean for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(par_name):
            self.mean[p][i][run] = par[j]


    def set_std(self, dpar, par_name, i, run):
        """Set std for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(par_name):
            self.std[p][i][run] = dpar[j]


    def plot_mean_std(self, n, n_D, par=[1, 2.5], par_name=['a', 'b'], out_name='mean_std'):
        """Plot mean and std versus number of realisations n

        Parameters
        ----------
        n: array of integer
            number of realisations {n_S} for ML covariance
        n_D: integer
            dimension of data vector
        par: array of float, optional
            input parameter values, default=[1, 2.5]
            input value for slope, default=2.5
        par_name: array of string, optional
            parameter names, default=['a', 'b']
        out_name: string, optional
            output file name base, default='mean_std'

        Returns
        -------
        None
        """

        marker = ['b.', 'bD']

        plot_sth = False
        plt.figure()
        plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R))

        for i, p in enumerate(par_name):
            if self.mean[p].any():
                plt.subplot(1, 2, 1)
                plt.plot(n, self.mean[p].mean(axis=1), marker[i])
                plt.plot([n[0], n[-1]], [par[i], par[i]], 'r-')
                plt.xlabel('n_S')
                plt.ylabel('<mean>')
                plot_sth = True

            if self.std[p].any():
                ax = plt.subplot(1, 2, 2)
                plt.plot(n, self.std[p].mean(axis=1), marker[i])
                plt.xlabel('n_S')
                plt.ylabel('<std>')
                ax.set_yscale('log')
                plot_sth = True

        if plot_sth == True:
            plt.savefig('{}.pdf'.format(out_name))



par_name = ['a', 'b']            # Parameter list
tr_name  = ['tr']

# Number of simulations
start = n_D + 3
stop  = n_D + 1250
n_S_arr = np.logspace(np.log10(start), np.log10(stop), 10, dtype='int')
#n_S_arr = np.arange(n_D+1, n_D+1250, 250)
n_n_S = len(n_S_arr)

# Number of runs per simulation
n_R = 10

# Results
sigma_ML    = Results(tr_name, n_n_S, n_R)
sigma_m1_ML = Results(tr_name, n_n_S, n_R)

fish_ana = Results(par_name, n_n_S, n_R)
fish_num = Results(par_name, n_n_S, n_R)
fish_deb = Results(par_name, n_n_S, n_R)
fit      = Results(par_name, n_n_S, n_R)


print('Creating {} simulations with {} runs each'.format(n_n_S, n_R))

# Go through number of simulations
for i, n_S in enumerate(n_S_arr):

    print('{}/{}: n_S={}'.format(i, n_n_S, n_S))

    n.append(n_S)                                             # number of data points

    # Loop over realisations
    for run in range(n_R):

        cov_est = get_cov_ML(yreal, cov, n_S)

        # Normalised trace
        this_sigma_ML = np.trace(cov_est) / n_D
        sigma_ML.set_mean([this_sigma_ML], tr_name, i, run)

        cov_est_inv = np.linalg.inv(cov_est)

        this_sigma_m1_ML = np.trace(cov_est_inv) / n_D
        sigma_m1_ML.set_mean([this_sigma_m1_ML], tr_name, i, run)

        ### Fisher matrix ###
        # analytical
        F = Fisher_ana(yreal, cov_est_inv)
        dpar = Fisher_error(F)
        fish_ana.set_std(dpar, par_name, i, run)

        # numerical
        F = Fisher_num(x1, a, b, cov_est_inv)
        dpar = Fisher_error(F)
        fish_num.set_std(dpar, par_name, i, run)

        # using debiased inverse covariance estimate
        cov_est_inv_debiased = debias_cov(cov_est_inv, n_S)
        F = Fisher_num(x1, a, b, cov_est_inv_debiased)
        dpar = Fisher_error(F)
        fish_deb.set_std(dpar, par_name, i, run)

        # MCMC fit of Parameters
        if do_fit_stan == True:
            res = fit_corr(x1, cov, cov_est, n_jobs=n_jobs)
            la  = res.extract(permuted=True)
            par  = []
            dpar = []
            for p in par_name:
                par.append(np.mean(la[p]))
                dpar.append(np.std(la[p]))
            fit.set_mean(par, par_name, i, run)
            fit.set_std(dpar, par_name, i, run)


plot_sigma_ML(n, n_D, sigma_ML.mean['tr'].mean(axis=1), sigma_m1_ML.mean['tr'].mean(axis=1), sig, out_name='sigma_ML')

fit.plot_mean_std(n, n_D, out_name='mean_std_MCMC', par=[a, b], par_name=par_name)
fish_num.plot_mean_std(n, n_D, out_name='std_Fisher_num', par=[a, b], par_name=par_name)
fish_ana.plot_mean_std(n, n_D, out_name='std_Fisher_ana', par=[a, b], par_name=par_name)
fish_deb.plot_mean_std(n, n_D, out_name='std_Fisher_deb', par=[a, b], par_name=par_name)


