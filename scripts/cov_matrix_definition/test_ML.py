import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys
import os
import re
import time
import copy
from astropy.table import Table, Column
from astropy.io import ascii
from optparse import OptionParser
from optparse import OptionGroup

"""
test_ML.py.

Example run to create simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m s  -r   -n   4   --n_n_S   10   -R 100

Add more simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m s  -r   -n   4   --n_n_S   10   -R 100 -a

Read existing simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m r  -r   -n   4   --n_n_S   10   -R 100
"""



import pdb

"""The following functions correspond to the expectation values for Gaussian
   distributions. See Taylor, Joachimi & Kitching (2013), TKB13
"""


def no_bias(n, n_D, par):
    """Unbiased estimator of par.
       For example maximum-likelihood estimate of covariance normalised trace,
       TKJ13 (17).
       Or Fisher matrix errors from debiased estimate of inverse covariance.
    """

    return np.asarray([par] * len(n))




def tr_N_m1_ML(n, n_D, par):
    """Maximum-likelihood estimate of inverse covariance normalised trace.
       TJK13 (24).
       This is alpha.
    """

    return [(n_S - 1.0)/(n_S - n_D - 2.0) * par for n_S in n]



def par_fish(n, n_D, par):
    """Fisher matrix parameter, not defined for mean.
       Expectation value of TJK13 (43), follows from (25).
       This is 1/sqrt(alpha).
    """

    return [np.sqrt((n_S - n_D - 2.0)/(n_S - 1.0)) * par for n_S in n]


def std_fish_deb(n, n_D, par):
    """Error on variance from Fisher matrix with debiased inverse covariance estimate.
       TJK13 (50, 55)
    """

    #return [np.sqrt(2.0 / (n_S - n_D - 4.0)) * par for n_S in n]
    return [np.sqrt(2 * A_corr(n_S, n_D) * (1.0 + (n_S - n_D - 2))) * par for n_S in n]



def A(n_S, n_D):
    """Return TJK13 (27)
    """

    A = (n_S - 1.0)**2 / ((n_S - n_D - 1.0) * (n_S - n_D - 2.0)**2 * (n_S - n_D - 4.0))

    return A


def A_corr(n_S, n_D):
    """Return TJK13 (28)
    """

    
    A_c =  1.0 / ((n_S - n_D - 1.0) * (n_S - n_D - 4.0))

    return A_c



def std_fish_biased(n, n_D, par):
    """Error on variance from Fisher matrix with biased inverse covariance estimate.
       From TJK13 (49) with A (27) instead of A_corr (28) in (49)
    """

    return [np.sqrt(2 * A(n_S, n_D) * (1.0 + (n_S - n_D - 2))) * par for n_S in n]


def d(x, sig2, delta):
    """Return determinant of Fisher matrix.
    """

    dp, det = Fisher_error_ana(x, sig2, delta, mode=-1)
    return det



def deltaG2(a, x, n_S, sig2, delta):
    """Return <(Delta G_aa)^2>.
    """

    n_D = len(x)

    if a==0:
        dG2 = 2 * A(n_S, n_D) / sig2**2 * (n_S - n_D - 1) * n_D**2
    elif a==1:
        dG2 = 2 * A(n_S, n_D) / sig2**2 * (n_S - n_D - 1) * (delta**2/12.0)**2
    else:
        error('Invalid parameter index {}'.format(a))

    return dG2



def std_fish_biased_ana(a, n, x, sig2, delta):

    return [np.sqrt(1.0/d(x, sig2, delta)**2 * (deltaG2(a, x, n_S, sig2, delta))) for n_S in n]



class Results:
    """Store results of sampling
    """

    def __init__(self, par_name, n_n_S, n_R, file_base='mean_std', yscale='linear', fct=None):
        """Set arrays for mean and std storing all n_S simulation cases
           with n_R runs each
        """

        self.mean      = {}
        self.std       = {}
        self.par_name  = par_name
        self.file_base = file_base

        if np.isscalar(yscale):
            self.yscale = [yscale, yscale]
        else:
            self.yscale = yscale

        self.fct       = fct     
        for p in par_name:
            self.mean[p]   = np.zeros(shape = (n_n_S, n_R))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R))


    def set(self, par, i, run, which='mean'):
        """Set mean or std for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(self.par_name):
            w = getattr(self, which)
            w[p][i][run] = par[j]


    def get_std_var(self, p):
        """Return standard deviation of the variance over all runs
        """

        n_n_S = self.mean[self.par_name[0]].shape[0]
        std_var = np.zeros(shape = n_n_S)
        for i in range(n_n_S):
            std_var[i] = np.std(self.std[p][i]**2)
            #std_var[i] = np.std(self.std[p][i]) # for testing

        return std_var


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


    def append(self, new, verbose=False):
        """Append new result to self.

        Parameters
        ----------
        self, new: class Result
            previous and new result
        verbose: bool, optional
            verbose mode

        Returns
        -------
        res: bool
            True for success
        """

        n_n_S, n_R         = self.mean[self.par_name[0]].shape
        n_n_S_new, n_R_new = new.mean[new.par_name[0]].shape
        if n_n_S != n_n_S_new:
            error( \
                'Number of simulations different for previous ({}) and new ({}), skipping append...'.format( \
                n_n_S, n_n_S_new), stop=False, verbose=verbose)

            return False

        for p in self.par_name:
            mean      = self.mean[p]
            std       = self.std[p]

            self.mean[p]   = np.zeros(shape = (n_n_S, n_R + n_R_new))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R + n_R_new))
            for n_S in range(n_n_S):
                self.mean[p][n_S]   = np.append(mean[n_S], new.mean[p][n_S])
                self.std[p][n_S]    = np.append(std[n_S], new.std[p][n_S])

        return True


    def plot_mean_std(self, n, n_D, par=None):
        """Plot mean and std versus number of realisations n

        Parameters
        ----------
        n: array of integer
            number of realisations {n_S} for ML covariance
        n_D: integer
            dimension of data vector
        par: dictionary of array of float, optional
            input parameter values and errors, default=None

        Returns
        -------
        None
        """

        n_R = self.mean[self.par_name[0]].shape[1]

        marker     = ['.', 'D']
        #markersize = [2] * len(marker)
        markersize = [6] * len(marker)
        color      = ['b', 'g']
        fac_xlim   = 1.05

        plot_sth = False
        plt.figure()
        plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R))

        box_width = (n[1] - n[0]) / 2

        # Set the number of required subplots (1 or 2)
        j_panel = {}
        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    n_panel = 2
                    j_panel[which] = j+1

        if len(j_panel) == 1:   # Only one plot to do: Use entire canvas
            n_panel = 1
            j_panel[j_panel.keys()[0]] = 1
        

        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    ax = plt.subplot(1, n_panel, j_panel[which])

                    if y.shape[1] > 1:
                        bplot = plt.boxplot(y.transpose(), positions=n, sym='.', widths=box_width)
                        for key in bplot:
                            plt.setp(bplot[key], color=color[i])
                        plt.setp(bplot['whiskers'], linestyle='-')
                    else:
                        plt.plot(n, y.mean(axis=1), marker[i], ms=markersize[i], color=color[i])

                    my_par = par[which]
                    if self.fct is not None and which in self.fct:
                        # Define high-resolution array for smoother lines
                        n_fine = np.arange(n[0], n[-1], len(n)/10.0)
                        plt.plot(n_fine, self.fct[which](n_fine, n_D, my_par[i]), '{}-.'.format(color[i]))

                    plt.plot(n, no_bias(n, n_D, my_par[i]), '{}-'.format(color[i]), label='{}$({})$'.format(which, p))

        # Finalize plot
        for j, which in enumerate(['mean', 'std']):
            if which in j_panel:
                ax = plt.subplot(1, n_panel, j_panel[which])
                plt.xlabel('$n_{\\rm s}$')
                plt.ylabel('<{}>'.format(which))
                #plt.xticks2()?bo alpha, or n_d / n_s
                plt.xlim((n[0]-5)/fac_xlim**3, n[-1]*fac_xlim)
                ax.set_yscale(self.yscale[j])
                plot_sth = True

                ax.legend(loc='lower right', frameon=False)

        if plot_sth == True:
            plt.savefig('{}.pdf'.format(self.file_base))


    def plot_std_var(self, n, n_D, par=None):
        """Plot standard deviation of parameter variance
        """

        n_R = self.mean[self.par_name[0]].shape[1]
        color = ['g', 'm']

        plot_sth = False
        plt.figure()
        plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R))
        ax = plt.subplot(1, 1, 1)

        for i, p in enumerate(self.par_name):
            y = self.get_std_var(p)
            if y.any():
                plt.plot(n, y, marker='o', color=color[i], label='$\sigma[\sigma^2({})]$'.format(p))

                if self.fct is not None and par is not None:
                    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
                    plt.plot(n_fine, self.fct['std_var'](n_fine, n_D, par[i]), '-', color=color[i])

                plt.xlabel('n_S')
                plt.ylabel('std(var)')
                plt.legend(loc='best', numpoints=1, frameon=False)
                ax.set_yscale('log')
                plot_sth = True

        plt.ylim(8e-9, 1e-2)

        if plot_sth == True:
            plt.savefig('std_2{}.pdf'.format(self.file_base))



def plot_std_fish_biased_ana(par_name, n, x, sig2, delta):

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    color = ['g', 'm']

    for i, p in enumerate(par_name):
        n_fine = np.arange(n[0], n[-1], len(n)/10.0)
        plt.plot(n_fine, std_fish_biased_ana(i, n_fine, x, sig2, delta), '-', color=color[i],
                 label='$\sigma[\sigma^2({})] t_1$ '.format(p))

    plt.xlabel('n_S')
    plt.ylabel('std(var)')
    plt.legend(loc='best', numpoints=1, frameon=False)
    ax.set_yscale('log')
    plt.ylim(8e-9, 1e-2)
    plt.savefig('{}.pdf'.format('std_var_ana'))


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
        #print>>f, ' ',

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
        print>>sys.stderr, "\x1b[31m{}\x1b[0m".format(str),

    if stop is False:
        if verbose is True:
            print>>sys.stderr,  "\x1b[31m{}, continuing...\x1b[0m".format(str),
    else:
        if verbose is True:
            print>>sys.stderr, ''
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
        n_n_S = 10,
        spar = '1.0 0.0',
        sig2 = 5.0,
        mode   = 's',
        do_fit_stan = False,
        do_fish_ana = False,
        likelihood  = 'norm',
        n_jobs = 1,
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
    parser.add_option('', '--n_n_S', dest='n_n_S', type='int', default=p_def.n_n_S,
        help='Number of n_S, where n_S is the number of simulation, default={}'.format(p_def.n_n_S))

    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='list of parameter values, default=\'{}\''.format(p_def.spar))
    parser.add_option('-s', '--sig2', dest='sig2', type='float', default=p_def.sig2,
        help='variance of Gaussian, default=\'{}\''.format(p_def.sig2))

    parser.add_option('', '--fit_stan', dest='do_fit_stan', action='store_true',
        help='Run stan for MCMC fitting, default={}'.format(p_def.do_fit_stan))
    parser.add_option('', '--fish_ana', dest='do_fish_ana', action='store_true',
        help='Calculate analytical Fisher matrix, default={}'.format(p_def.do_fish_ana))
    parser.add_option('-L', '--like', dest='likelihood', type='string', default=p_def.likelihood,
        help='Likelihoo for MCMC, one in \'norm\'|\'ST\', default=\'{}\''.format(p_def.likelihood))

    parser.add_option('-m', '--mode', dest='mode', type='string', default=p_def.mode,
        help='Mode: \'s\'=simulate, \'r\'=read, default={}'.format(p_def.mode))
    parser.add_option('-a', '--add_simulations', dest='add_simulations', action='store_true', help='add simulations to existing files')
    parser.add_option('-r', '--random_seed', dest='random_seed', action='store_true', help='random seed')

    parser.add_option('-n', '--n_jobs', dest='n_jobs', type='int', default=p_def.n_jobs,
        help='Number of parallel jobs, default={}'.format(p_def.n_jobs))

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

    if options.add_simulations == True:
        if options.random_seed is False:
            error('Adding simulations (-a) only possible with random seed (-r)')
        if re.search('r', options.mode) is not None:
            error('Adding simulations (-a) is not  possible in read mode (-m r)')

    if re.search('s', options.mode) is not None and re.search('r', options.mode) is not None:
        error('Simulation and read mode (-m rs) not possible simultaneously') 

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



def numbers_from_file(file_base, npar):
    """Return number of simulation and runs as read from simulation file"""

    dat = ascii.read('{}.txt'.format(file_base))

    # Number of simulation cases = number of rows in file
    n_n_S = len(dat['n_S'])

    # Number of runs: number of columns, subtract one (n_S), divide by 2 (mean+std),
    # devide by number of parameters
    n_R   = (len(dat.keys()) - 1) / 2 / npar

    return n_n_S, n_R



# Gaussian likelihood function
# chi^2 = -2 log L = (y - mu)^t Psi (y - mu).
# y: n_D dimensional data vector, simulated as y(x) ~ N(mu(x), C)
#    with mu(x) = b + a * x
# x: x ~ Uniform(-100, 100)
# mu: mean, mu(x) = b + a * x, with parameters b, a
# Psi: estimated inverse covariance, Phi^-1 times correction factor
# Phi: estimate of true covariance C, ML estimate from n_S realisations of y.
# C = diag(sig^2, ..., sig^2)
# Fisher matrix
# F_rs = 1/2 ( dmu^t/dr Psi dmu/ds + dmu^t/ds Psi dmu/dr)
#      = 1/2 |2 x^t Psi x               x^t Psi 1 + 1^t Psi x| 
#            |1^t Psi x + x^t Psi 1     2 1^t Psi 1|
#      =     |x^t Psi x   x^t Psi 1|
#            |x^t Psi 1   1^1 Psi 1|
#      =     |sum_ij x_i Psi_ij x_j   sum_ij x_i Psi_ij|
#            |sum_ij x_i Psi_ij       sum_ij Psi_ij|
# e.g. F_11 = F_aa = x Psi x^t

def Fisher_ana_ele(r, s, x, Psi):
    """Return analytical Fisher matrix element (r, s).

    Parameters
    ----------
    r, s: integer
        indices of matrix, r,s in {0,1}
    x: array of float
        abcissa for data vector y(x)=a*x+b
    Psi: matrix
        precision matrix

    Returns
    -------
    f_rs: float
        Fisher matrix element (r, s)
    """

    n_D = len(x)
    v = np.zeros(shape = (2, n_D))
    for i in (r, s):
        if i == 0:
            v[i] = x
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
        dy_dr = ((a+h) * x + b - ((a-h) * x + b)) / (2*h)
    else:
        h = 0.1
        dy_dr = (a * x + b+h - (a * x + b-h)) / (2*h)

    if s == 0:
        h = 0.1 * a
        dy_ds = ((a+h) * x + b - ((a-h) * x + b)) / (2*h)
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
    d    = np.sqrt(np.diag(Finv))

    return d


def Fisher_error_ana(x, sig2, delta, mode=-1):
    """Return Fisher matrix errors, and detminant if mode==2, for affine function parameters (a, b)
    """

    n_D = len(x)

    # The three following ways to compute the Fisher matrix errors are equivalent.
    # Note that mode==-1,0 uses the statistical properties mean and variance of the uniform
    # distribution, whereas more=1,2 uses the actual sample x.

    if mode != -1:
        if mode == 2:
            Psi = np.diag([1.0 / sig2 for i in range(n_D)])
            F_11 = np.einsum('i,ij,j', x, Psi, x)
            F_22 = np.einsum('i,ij,j', np.ones(shape=n_D), Psi, np.ones(shape=n_D))
            F_12 = np.einsum('i,ij,j', x, Psi, np.ones(shape=n_D))
        elif mode == 1:
            F_11 = sum(x*x) / sig2
            F_12 = sum(x) / sig2
            F_22 = n_D / sig2
        elif mode == 0:
            F_11 = n_D * delta**2 / 12.0 / sig2
            F_12 = 0 
            F_22 = n_D / sig2

        det = F_11 * F_22 - F_12**2
        da2 = F_22 / det
        db2 = F_11 / det
    else:
        # mode=-1
        det = (n_D/sig2)**2 * delta**2/12
        da2 = 12 * sig2 / (n_D * delta**2)
        db2 = sig2 / n_D

    return np.sqrt([da2, db2]), det


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



def plot_sigma_ML(n, n_D, sigma_ML, sigma_m1_ML, sig2, out_name='sigma_ML'):
    """Obsolete, use class method instead.
    """

    plt.figure()
    plt.suptitle('Covariance of n_D = {} dimensional data'.format(n_D))

    plt.subplot(1, 2, 1)
    plt.plot(n, sigma_ML, 'b.')
    plt.plot([n[0], n[-1]], [sig2, sig2], 'r-')
    plt.xlabel('n_S')
    plt.ylabel('normalised trace of ML covariance')

    ax = plt.subplot(1, 2, 2)
    plt.plot(n, sigma_m1_ML, 'b.')
    plt.plot([n[0], n[-1]], [1.0/sig2, 1.0/sig2], 'r-')
    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
    bias = [(n_S-1.0)/(n_S-n_D-2.0)/sig2 for n_S in n_fine]
    plt.plot(n_fine, bias, 'g-.')
    plt.xlabel('n_S')
    plt.ylabel('normalised trace of inverse of ML covariance')
    ax.set_yscale('log')

    plt.savefig('{}.pdf'.format(out_name))

    f = open('{}.txt'.format(out_name), 'w')
    print >>f, '# sig2={}, n_D={}'.format(sig2, n_D)
    print >>f, '# n sigma 1/sigma'
    for i in range(len(n)):
        print >>f, '{} {} {}'.format(n[i], sigma_ML[i], sigma_m1_ML[i])
    f.close()



def fit_corr(x1, cov_true, cov_est, n_jobs=3):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_est, square covariance matrix for the fitting
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
    toy_data['cov_est'] = cov_est

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

    import pystan
    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=3, verbose=False, n_jobs=n_jobs)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit


def fit_corr_ST(x1, cov_true, cov_est_inv, n_jobs=3):
    """
    Generates one draw from a multivariate student-t distribution
    (see Sellentin & Heavens 2015)
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_est_inv, inverse square covariance matrix for the fitting
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

    # set estimated inverse covariance matrix for fitting
    toy_data['cov_est_inv'] = cov_est_inv

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs;                                 
        vector[nobs] x;                       
        vector[nobs] y;   
        matrix[nobs, nobs] cov_est_inv;                    
    }
    parameters {
        real a;
        real b;                                                              
    }
    model {
        vector[nobs] mu;
        real chi2;

        mu = b + a * x;

        # Sellentin & Heavens debiasing scheme:
        # Replace normal likelihood with t-distribution
        # y ~ (1 + (log_normal(mu, cov_est)) / (1 + n_S))^(-n_S/2)  

        chi2 = (y - mu)' * cov_est_inv * (y - mu);

        target += pow(1.0 + chi2/(1.0 + nobs), -nobs/2.0);
    }
    """


    import pystan
    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=3, verbose=False, n_jobs=n_jobs)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit



def simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_G, fit_ST, options, verbose=False):
    """Simulate data"""
        
    if verbose == True:
        print('Creating {} simulations with {} runs each'.format(len(n_S_arr), options.n_R))

    cov = np.diag([options.sig2 for i in range(options.n_D)])            # *** cov of the data in the same catalog! ***

    # Go through number of simulations
    for i, n_S in enumerate(n_S_arr):

        if verbose == True:
            print('{}/{}: n_S={}'.format(i+1, len(n_S_arr), n_S))

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
                F = Fisher_ana(x1, cov_est_inv)   # Bug fix 05/10, 1st arg should be x1, not yreal
                dpar = Fisher_error(F)
                fish_ana.set(dpar, i, run, which='std')

            # numerical
            F = Fisher_num(x1, options.a, options.b, cov_est_inv)
            dpar = Fisher_error(F)
            fish_num.set(dpar, i, run, which='std')

            # The following also works, if get_std_var is changed at same time
            #fish_num.set(dpar**2, i, run, which='std')

            # using debiased inverse covariance estimate
            cov_est_inv_debiased = debias_cov(cov_est_inv, n_S)
            F = Fisher_num(x1, options.a, options.b, cov_est_inv_debiased)
            dpar = Fisher_error(F)
            fish_deb.set(dpar, i, run, which='std')

            # MCMC fit of Parameters
            if options.do_fit_stan == True:
                if options.likelihood == 'norm':
                    if verbose == True:
                        print('Running MCMC with mv normal likelihood')
                    res = fit_corr(x1, cov, cov_est, n_jobs=options.n_jobs)
                    fit = fit_norm
                elif options.likelihood == 'ST':
                    if verbose == True:
                        print('Running MCMC with Sellentin&Heavens (ST) likelihood')
                    res = fit_corr_ST(x1, cov, cov_est_inv, n_jobs=options.n_jobs)
                    fit = fit_ST
                else:
                    error('Invalid likelihood \'{}\''.format(options.likelihood))

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

    if options.add_simulations == True:
        if verbose == True:
            print('Reading previous simulations from disk')

        # Initialise results
        n_n_S, n_R  = numbers_from_file(sigma_ML.file_base, 1)
        #sigma_ML_prev    = Results(sigma_ML.par_name, n_n_S, n_R, file_base=sigma_ML.file_base, fct=sigma_ML.fct)
        #sigma_m1_ML_prev = Results(sigma_m1_ML.par_name, n_n_S, n_R, file_base=sigma_m1_ML.file_base, yscale='log', fct=sigma_m1_ML.fct)
        sigma_ML_prev    = Results(sigma_ML.par_name, n_n_S, n_R, file_base=sigma_ML.file_base)
        sigma_m1_ML_prev = Results(sigma_m1_ML.par_name, n_n_S, n_R, file_base=sigma_m1_ML.file_base)
        fish_ana_prev    = Results(fish_ana.par_name, n_n_S, n_R, file_base=fish_ana.file_base, fct=fish_ana.fct)
        fish_num_prev    = Results(fish_num.par_name, n_n_S, n_R, file_base=fish_num.file_base, fct=fish_num.fct, yscale='linear')
        fish_deb_prev    = Results(fish_deb.par_name, n_n_S, n_R, file_base=fish_deb.file_base, fct=fish_deb.fct)
        fit_prev         = Results(fit.par_name, n_n_S, n_R, file_base=fit.file_base, fct=fit.fct)
        # Fill results from files
        read_from_file(sigma_ML_prev, sigma_m1_ML_prev, fish_ana_prev, fish_num_prev, fish_deb_prev, fit_prev, options, verbose=options.verbose)

        # Add new results
        sigma_ML.append(sigma_ML_prev)
        sigma_m1_ML.append(sigma_m1_ML_prev)
        fish_ana.append(fish_ana_prev)
        fish_num.append(fish_num_prev)
        fish_deb.append(fish_deb_prev)
        fit.append(fit_prev)

    if verbose == True:
        print('Writing simulations to disk')

    if options.do_fish_ana == True:
        fish_ana.write_mean_std(n_S_arr)
    fish_num.write_mean_std(n_S_arr)
    fish_deb.write_mean_std(n_S_arr)
    if options.do_fit_stan == True:
        fit.write_mean_std(n_S_arr)

    sigma_ML.write_mean_std(n_S_arr)
    sigma_m1_ML.write_mean_std(n_S_arr)



def read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_ST, options, verbose=False):
    """Read simulated runs from files"""

    if verbose == True:
        print('Reading simulations from disk')

    if options.do_fish_ana == True:
        fish_ana.read_mean_std()
    fish_num.read_mean_std()
    fish_deb.read_mean_std()

    if options.do_fit_stan:
        fit_norm.read_mean_std()
        fit_ST.read_mean_std()

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

    par_name = ['a', 'b']            # Parameter list
    tr_name  = ['tr']                # For cov plots
    delta    = 200                   # Width of uniform distribution for x

    # Number of simulations
    start = options.n_D + 5
    stop  = options.n_D * 10
    n_S_arr = np.logspace(np.log10(start), np.log10(stop), options.n_n_S, dtype='int')
    n_n_S = len(n_S_arr)


    # Initialisation of results
    sigma_ML    = Results(tr_name, n_n_S, options.n_R, file_base='sigma_ML')
    sigma_m1_ML = Results(tr_name, n_n_S, options.n_R, file_base='sigma_m1_ML', yscale='log', fct={'mean': tr_N_m1_ML})

    fish_ana = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_ana', yscale='log', fct={'std': par_fish})
    fish_num = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_num', yscale='log', fct={'std': par_fish, 'std_var': std_fish_biased})
    fish_deb = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_deb', yscale='log', fct={'std': no_bias, 'std_var': std_fish_deb})
    fit_norm = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_norm', yscale=['linear', 'log'],
                       fct={'std': par_fish, 'std_var': std_fish_biased})
    fit_ST   = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_ST', yscale=['linear', 'log'],
                       fct={'std': par_fish, 'std_var': std_fish_biased})

    # Data
    x1 = uniform.rvs(loc=-delta/2, scale=delta, size=options.n_D)        # exploratory variable
    x1.sort()
    yreal = options.a * x1 + options.b

    # Create simulations
    if re.search('s', options.mode) is not None:
 
        simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_ST, options, verbose=options.verbose) 

        write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_ST, options, verbose=options.verbose)


    # Read simulations
    if re.search('r', options.mode) is not None:

        read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_ST, options, verbose=options.verbose)


    # Plot results. Obsolete function, now done by class routine.
    #plot_sigma_ML(n_S_arr, options.n_D, sigma_ML.mean['tr'].mean(axis=1), sigma_m1_ML.mean['tr'].mean(axis=1), options.sig2, out_name='sigma_both')

    dpar_exact, det = Fisher_error_ana(x1, options.sig2, delta, mode=-1)

    # Exact inverse covariance
    #cov_inv    = np.diag([1.0 / options.sig2 for i in range(options.n_D)])
    #F_exact    = Fisher_ana(x1, cov_inv)    # Bug fixed 05/10
    #dpar_exact = Fisher_error(F_exact)
    #print('Parameter error (from exact Fisher_ana)', dpar_exact)

    if options.verbose == True:
        print('Creating plots')

    if options.do_fish_ana == True:
        fish_ana.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})
    fish_num.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})
    fish_deb.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})
    if options.do_fit_stan == True:
        fit_norm.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact})
        fit_ST.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact})

    dpar2 = dpar_exact**2

    # Problem with this plot, simulation does not agree with analytical formula.
    # Checked:
    # 08/09/2017
    # - std**2 = var, seems to be correct in the code.
    # To check (again): Does points go -> 0 for n_S very large or stay constant?
    # Could be higher-order effect at low n_s?
    fish_num.plot_std_var(n_S_arr, options.n_D, par=dpar2)

    fish_deb.plot_std_var(n_S_arr, options.n_D, par=dpar2)
    if options.do_fit_stan == True:
        fit_norm.plot_std_var(n_S_arr, options.n_D, par=dpar2)
        fit_ST.plot_std_var(n_S_arr, options.n_D, par=dpar2)

    sigma_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [options.sig2]})
    sigma_m1_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [1/options.sig2]})

    plot_std_fish_biased_ana(par_name, n_S_arr, x1, options.sig2, delta)

    ### End main program

    if options.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

