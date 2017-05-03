import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys
import os
import re
import pystan
import time
import copy
from astropy.table import Table, Column
from astropy.io import ascii
from optparse import OptionParser
from optparse import OptionGroup

"""The following functions correspond to the expectation values for Gaussian
   distributions. See Taylor, Joachimi & Kitching (2013), TKB13
"""


def no_bias(n, n_D, par):
    """Unbiased estimator of par.
       For example maximum-likelihood estimate of covariance normalised trace,
       TKJ13 (17).
    """

    return np.asarray([par] * len(n))



def tr_N_m1_ML(n, n_D, par):
    """Maximum-likelihood estimate of inverse covariance normalised trace.
       TJK13 (24)
    """

    return [(n_S - 1.0)/(n_S - n_D - 2.0)/par for n_S in n]



def par_fish(n, n_D, par):
    """Fisher matrix parameter, not defined for mean.
       TJK13 (55)
    """

    return [np.sqrt((n_S - n_D - 2.0)/(n_S - 1.0))*par for n_S in n]


def std_fish(n, n_D, par):
    """Fisher matrix error on error.
       TJK13 (55)
    """

    return [np.sqrt(2.0 / (n_S - n_D - 4.0)) * par for n_S in n]



class Results:
    """Store results of sampling
    """

    def __init__(self, par_name, n_n_S, n_R, file_base='mean_std', yscale='linear', fct=None):
        """Set arrays for mean and std storing all n_S simulation cases
           with n_R runs each
        """

        self.mean      = {}
        self.std       = {}
        self.stdstd    = {}
        self.par_name  = par_name
        self.file_base = file_base
        self.yscale    = yscale
        self.fct       = fct     
        for p in par_name:
            self.mean[p]   = np.zeros(shape = (n_n_S, n_R))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R))
            self.stdstd[p] = np.zeros(shape = (n_n_S))


    def set(self, par, i, run, which='mean'):
        """Set mean or std for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(self.par_name):
            w = getattr(self, which)
            w[p][i][run] = par[j]


    def set_stdstd(self):
        """Set standard deviation of standard deviation over all runs
        """

        n_n_S = self.mean[self.par_name[0]].shape[0]
        for p in self.par_name:
            for i in range(n_n_S):
                self.stdstd[p][i] = np.std(self.std[p][i])


    def read_mean_std(self, format='ascii'):
        """Read mean and std from file
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            dat = ascii.read('{}.txt'.format(self.file_base))
            for p in self.par_name:
                for run in range(n_R):
                    col_name = 'mean[{0:s}]_run{1:02d}'.format(p, run)
                    self.mean[p].transpose()[run] = dat[col_name]
                    col_name = 'std[{0:s}]_run{1:02d}'.format(p, run)
                    self.std[p].transpose()[run] = dat[col_name]


    def write_mean_std(self, n, format='ascii'):
        """Write mean and std to file
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii': 
            cols  = [n]
            names = ['# n_S']
            for p in self.par_name:
                for run in range(n_R):
                    cols.append(self.mean[p].transpose()[run])
                    names.append('mean[{0:s}]_run{1:02d}'.format(p, run))
                    cols.append(self.std[p].transpose()[run])
                    names.append('std[{0:s}]_run{1:02d}'.format(p, run))
            t = Table(cols, names=names)
            f = open('{}.txt'.format(self.file_base), 'w')
            ascii.write(t, f, delimiter='\t')
            f.close()



    def plot_mean_std(self, n, n_D, par=[1, 2.5]):
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

        Returns
        -------
        None
        """

        n_R = self.mean[self.par_name[0]].shape[1]

        marker     = ['.', 'D']
        markersize = [2] * len(marker)
        color      = ['b', 'g']
        fac_xlim   = 1.05

        plot_sth = False
        plt.figure()
        plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R))

        box_width = (n[1] - n[0]) / 2   # Shouldn't really make sense for log-points in x, but seems to work anyway...

        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    ax = plt.subplot(1, 2, j+1)
                    plt.plot(n, y.mean(axis=1), marker[i], ms=markersize[i], color=color[i])

                    if y.shape[1] > 1:
                        plt.boxplot(y.transpose(), positions=n, sym='.', widths=box_width)

                    if self.fct is not None:
                        n_fine = np.arange(n[0], n[-1], len(n)/10.0)
                        plt.plot(n_fine, self.fct[which](n_fine, n_D, par[i]), 'g-.')

                    plt.xlabel('n_S')
                    plt.ylabel('<{}>'.format(which))
                    plt.xlim(n[0]/fac_xlim, n[-1]*fac_xlim)
                    ax.set_yscale(self.yscale)
                    plot_sth = True

        if plot_sth == True:
            plt.savefig('{}.pdf'.format(self.file_base))


    def plot_stdstd(self, n, n_D, par=None):
        """Plot standard deviation of standard deviation.
        """

        n_R = self.mean[self.par_name[0]].shape[1]
        color = ['g', 'm']

        plot_sth = False
        plt.figure()
        plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R))
        ax = plt.subplot(1, 1, 1)

        for i, p in enumerate(self.par_name):
            y = self.stdstd[p]
            if y.any():
                plt.plot(n, y, marker='o', color=color[i])

                if self.fct is not None and par is not None:
                    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
                    plt.plot(n_fine, self.fct['stdstd'](n_fine, n_D, par[i]), '-', color=color[i])

                plt.xlabel('n_S')
                plt.ylabel('std(std)')
                ax.set_yscale('log')
                plot_sth = True

        if plot_sth == True:
            plt.savefig('std_{}.pdf'.format(self.file_base))

class param:
    """General class to store (default) variables
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self, **kwds):
        print(self.__dict__)

    def var_list(self, **kwds):
        return vars(self)


def my_string_split(string, num=-1, verbose=False, stop=False):
    """Split a *string* into a list of strings. Choose as separator
        the first in the list [space, underscore] that occurs in the string.
        (Thus, if both occur, use space.)

    Parameters
    ----------
    string: string
        Input string
    num: int
        Required length of output list of strings, -1 if no requirement.
    verbose: bool
        Verbose output
    stop: bool
        Stop programs with error if True, return None and continues otherwise

    Returns
    -------
    list_str: string, array()
        List of string on success, and None if failed.
    """

    if string is None:
        return None

    has_space      = string.find(' ')
    has_underscore = string.find('_')

    if has_space != -1:
        # string has white-space
        sep = ' '
    else:
        if has_underscore != -1:
        # string has no white-space but underscore
            sep = '_'
        else:
            # string has neither, consists of one element
            if num == -1 or num == 1:
                # one-element string is ok
                sep = None
                pass
            else:
                error('Neither \' \' nor \'_\' found in string \'{}\', cannot split'.format(string))

    #res = string.split(sep=sep) # python v>=3?
    res = string.split(sep)

    if num != -1 and num != len(res):
        if verbose:
            print >>std.styerr, 'String \'{}\' has length {}, required is {}'.format(string, len(res), num)
        if stop is True:
            sys.exit(2)
        else:
            return None

    return res



def log_command(argv, name=None, close_no_return=True):
    """Write command with arguments to a file or stdout.
       Choose name = 'sys.stdout' or 'sys.stderr' for output on sceen.

    Parameters
    ----------
    argv: array of strings
        Command line arguments
    name: string
        Output file name (default: 'log_<command>')
    close_no_return: bool
        If True (default), close log file. If False, keep log file open
        and return file handler

    Returns
    -------
    log: filehandler
        log file handler (if close_no_return is False)
    """

    if name is None:
        name = 'log_' + os.path.basename(argv[0])

    if name == 'sys.stdout':
        f = sys.stdout
    elif name == 'sys.stderr':
        f = sys.stderr
    else:
        f = open(name, 'w')

    for a in argv:

        # Quote argument if special characters
        if ']' in a or ']' in a:
            a = '\"{}\"'.format(a)

        print>>f, a,
        print>>f, ' ',

    print>>f, ''

    if close_no_return == False:
        return f

    if name != 'sys.stdout' and name != 'sys.stderr':
        f.close()



def error(str, val=1, stop=True, verbose=True):
    """Print message str and exits program with code val

    Parameters
    ----------
    str: string
        message
    val: integer
        exit value, default=1
    stop: boolean
        stops program if True (default), continues if False
    verbose: boolean
        verbose output if True (default)

    Returns
    -------
    None
    """

    if verbose is True:
        print_color('red', str, file=sys.stderr, end='')

    if stop is False:
        if verbose is True:
            print_color('red', ', continuing', file=sys.stderr)
    else:
        if verbose is True:
            print_color('red', '', file=sys.stderr)
        sys.exit(val)



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class mkstuff.param
        parameter values
    """

    p_def = param(
        n_D = 750,
        n_R = 10,
        spar = '1.0 0.0',
        sig = 5.0,
        do_fit_stan = False,
        do_fish_ana = False,
        n_jobs = 1,
        mode   = 's',
        random_seed = True,
    )

    return p_def



def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class mkstuff.param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option('-D', '--n_D', dest='n_D', type='int', default=p_def.n_D,
        help='Number of data points, default={}'.format(p_def.n_D))
    parser.add_option('-R', '--n_R', dest='n_R', type='int', default=p_def.n_R,
        help='Number of runs per simulation, default={}'.format(p_def.n_R))
    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='list of parameter values, default=\'{}\''.format(p_def.spar))
    parser.add_option('-s', '--sig', dest='sig', type='float', default=p_def.sig,
        help='standard deviation of Gaussian, default=\'{}\''.format(p_def.sig))
    parser.add_option('', '--fit_stan', dest='do_fit_stan', action='store_true',
        help='Run stan for MCMC fitting, default={}'.format(p_def.do_fit_stan))
    parser.add_option('', '--fish_ana', dest='do_fish_ana', action='store_true',
        help='Calculate analytical Fisher matrix, default={}'.format(p_def.do_fish_ana))
    parser.add_option('-n', '--n_jobs', dest='n_jobs', type='int', default=p_def.n_jobs,
        help='Number of parallel jobs, default={}'.format(p_def.n_jobs))
    parser.add_option('-m', '--mode', dest='mode', type='string', default=p_def.mode,
        help='Mode: \'s\'=simulate, \'r\'=read, default={}'.format(p_def.mode))
    parser.add_option('-r', '--random_seed', dest='random_seed', action='store_true', help='random seed')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')

    options, args = parser.parse_args()

    return options, args



def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    see_help = 'See option \'-h\' for help.'

    return True



def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class mkstuff.param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class mkstuff.param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    # Do extra stuff if necessary
    par = my_string_split(param.spar, num=2, verbose=options.verbose, stop=True)
    options.par = [float(p) for p in par]
    options.a = options.par[0]
    options.b = options.par[1]

    return param



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
    """Obsolete, use class method instead.
    """

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
    ax.set_yscale('log')

    plt.savefig('{}.pdf'.format(out_name))

    f = open('{}.txt'.format(out_name), 'w')
    print >>f, '# sig={}, n_D={}'.format(sig, n_D)
    print >>f, '# n sigma 1/sigma'
    for i in range(len(n)):
        print >>f, '{} {} {}'.format(n[i], sigma_ML[i], sigma_m1_ML[i])
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


def simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=False):
    """Simulate data"""
        
    if verbose == True:
        print('Creating {} simulations with {} runs each'.format(len(n_S_arr), options.n_R))

    cov = np.diag([options.sig for i in range(options.n_D)])            # *** cov of the data in the same catalog! ***

    # Go through number of simulations
    for i, n_S in enumerate(n_S_arr):

        if verbose == True:
            print('{}/{}: n_S={}'.format(i, len(n_S_arr), n_S))

        # Loop over realisations
        for run in range(options.n_R):

            cov_est = get_cov_ML(yreal, cov, n_S)

            # Normalised trace
            this_sigma_ML = np.trace(cov_est) / options.n_D
            sigma_ML.set([this_sigma_ML], i, run, which='mean')

            cov_est_inv = np.linalg.inv(cov_est)

            this_sigma_m1_ML = np.trace(cov_est_inv) / options.n_D
            sigma_m1_ML.set([this_sigma_m1_ML], i, run, which='mean')

            ### Fisher matrix ###
            # analytical
            if options.do_fish_ana == True:
                F = Fisher_ana(yreal, cov_est_inv)
                dpar = Fisher_error(F)
                fish_ana.set(dpar, i, run, which='std')

            # numerical
            F = Fisher_num(x1, options.a, options.b, cov_est_inv)
            dpar = Fisher_error(F)
            fish_num.set(dpar, i, run, which='std')

            # using debiased inverse covariance estimate
            cov_est_inv_debiased = debias_cov(cov_est_inv, n_S)
            F = Fisher_num(x1, options.a, options.b, cov_est_inv_debiased)
            dpar = Fisher_error(F)
            fish_deb.set(dpar, i, run, which='std')

            # MCMC fit of Parameters
            if options.do_fit_stan == True:
                res = fit_corr(x1, cov, cov_est, n_jobs=options.jobs)
                la  = res.extract(permuted=True)
                par  = []
                dpar = []
                for p in fit.par_name:
                    par.append(np.mean(la[p]))
                    dpar.append(np.std(la[p]))
                fit.set(par, i, run, which='mean')
                fit.set(dpar, i, run, which='std')



def write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=False):
    """Write simulated runs to files"""

    if verbose == True:
        print('Writing simulations to disk')

    if options.do_fish_ana == True:
        fish_ana.write_mean_std(n_S_arr)
    fish_num.write_mean_std(n_S_arr)
    fish_deb.write_mean_std(n_S_arr)
    if options.do_fit_stan == True:
        fit.write_mean_std(n_S_arr, par=options.par)

    sigma_ML.write_mean_std(n_S_arr)
    sigma_m1_ML.write_mean_std(n_S_arr)



def read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=False):
    """Read simulated runs from files"""

    if verbose == True:
        print('Reading simulations from disk')

    if options.do_fish_ana == True:
        fish_ana.read_mean_std()
    fish_num.read_mean_std()
    fish_deb.read_mean_std()

    if options.do_fit_stan:
        fit.read_mean_std()

    sigma_ML.read_mean_std()
    sigma_m1_ML.read_mean_std()

    


# Main program
def main(argv=None):
    """Main program.
    """


    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)


    # Save calling command
    log_command(argv)
    if options.verbose:
        log_command(argv, name='sys.stderr')


    if options.verbose is True:
        print('Start program {}'.format(os.path.basename(argv[0])))


    ### Start main program ###

    if options.random_seed is False:
        np.random.seed(1056)                 # set seed to replicate example

    sigma_ML     = []
    sigma_m1_ML  = []


    par_name = ['a', 'b']            # Parameter list
    tr_name  = ['tr']                # For cov plots

    # Number of simulations
    start = options.n_D + 5
    stop  = options.n_D * 2
    n_S_arr = np.logspace(np.log10(start), np.log10(stop), 10, dtype='int')
    #n_S_arr = np.arange(n_D+1, n_D+1250, 250)
    n_n_S = len(n_S_arr)


    # Initialisation of results
    sigma_ML    = Results(tr_name, n_n_S, options.n_R, file_base='sigma_ML', fct={'mean': no_bias})
    sigma_m1_ML = Results(tr_name, n_n_S, options.n_R, file_base='sigma_m1_ML', yscale='log', fct={'mean': tr_N_m1_ML})

    fish_ana = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_ana', fct={'std': par_fish})
    fish_num = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_num', fct={'std': par_fish}, yscale='linear')
    fish_deb = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_deb', fct={'std': no_bias, 'stdstd': std_fish})
    fit      = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit')

    # Data
    x1 = uniform.rvs(loc=-100, scale=200, size=options.n_D)        # exploratory variable
    x1.sort()
    yreal = options.a * x1 + options.b

    # Create simulations
    if re.search('s', options.mode) is not None:
 
        simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=options.verbose) 

        write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=options.verbose)


    # Read simulations
    if re.search('r', options.mode) is not None:

        read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit, options, verbose=options.verbose)


    # Plot results
    plot_sigma_ML(n_S_arr, options.n_D, sigma_ML.mean['tr'].mean(axis=1), sigma_m1_ML.mean['tr'].mean(axis=1), options.sig, out_name='sigma_both')

    # Exact inverse covariance
    cov_inv    = np.diag([1.0 / options.sig for i in range(options.n_D)])
    F_exact    = Fisher_ana(yreal, cov_inv)
    dpar_exact = Fisher_error(F_exact)
    print('dpar_exact = ', dpar_exact)

    if options.do_fish_ana == True:
        fish_ana.plot_mean_std(n_S_arr, options.n_D, par=dpar_exact)
    fish_num.plot_mean_std(n_S_arr, options.n_D, par=dpar_exact)
    fish_deb.plot_mean_std(n_S_arr, options.n_D, par=dpar_exact)
    if options.do_fit_stan == True:
        fit.plot_mean_std(n_S_arr, options.n_D, par=options.par)

    fish_num.set_stdstd()
    fish_num.plot_stdstd(n_S_arr, options.n_D)

    fish_deb.set_stdstd()
    fish_deb.plot_stdstd(n_S_arr, options.n_D, par=dpar_exact)

    sigma_ML.plot_mean_std(n_S_arr, options.n_D, par=[options.sig])
    sigma_m1_ML.plot_mean_std(n_S_arr, options.n_D, par=[options.sig])

    ### End main program

    if options.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

