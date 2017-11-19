#!/usr/bin/env python

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

sys.path.append('../..')
from covest import *


"""
test_ML.py.

Example run to create simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m s  -r   -n   4   --n_n_S   10   -R 100

Add more simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m s  -r   -n   4   --n_n_S   10   -R 100 -a

Read existing simulations:
test_ML.py   -D 750   -p   1_0   -s   5   -v   -m r  -r   -n   4   --n_n_S   10   -R 100
"""



"""The following functions correspond to the expectation values for Gaussian
   distributions. See Taylor, Joachimi & Kitching (2013), TKB13
"""



def alpha(n_S, n_D):
    """Return precision matrix estimate bias prefactor alpha.
       IK17 (5).
    """

    return (n_S - 1.0)/(n_S - n_D - 2.0)



def A(n_S, n_D):
    """Return TJK13 (27)
    """

    A = alpha(n_S, n_D)**2 / ((n_S - n_D - 1.0) * (n_S - n_D - 4.0))

    return A



def A_corr(n_S, n_D):
    """Return TJK13 (28), this is A/alpha^2.
    """

    
    A_c =  1.0 / ((n_S - n_D - 1.0) * (n_S - n_D - 4.0))

    return A_c



def tr_N_m1_ML(n, n_D, par):
    """Maximum-likelihood estimate of inverse covariance normalised trace.
       TJK13 (24), IK17 (4).
       This is alpha.
    """

    return [alpha(n_S, n_D) * par for n_S in n]



def par_fish(n, n_D, par):
    """Fisher matrix parameter, not defined for mean.
       Expectation value of TJK13 (43), follows from (25).
       This is 1/sqrt(alpha).
    """

    return [np.sqrt(1.0 / alpha(n_S, n_D)) * par for n_S in n]


def std_fish_deb(n, n_D, par):
    """Error on variance from Fisher matrix with debiased inverse covariance estimate.
       TJK13 (49, 50)
    """

    #return [np.sqrt(2 * A_corr(n_S, n_D) * (n_S - n_D - 1)) * par for n_S in n]    # checked
    #return [np.sqrt(2 * A(n_S, n_D) / alpha(n_S, n_D)**2 * (n_S - n_D - 1)) * par for n_S in n]    # checked

    return [np.sqrt(2.0 / (n_S - n_D - 4.0)) * par for n_S in n]



def std_fish_biased(n, n_D, par):
    """Error on variance from Fisher matrix with biased inverse covariance estimate,
       with correction, IK17 (26).
    """

    return [np.sqrt(2 * A(n_S, n_D) * ((n_S - n_D - 1))) / (2 * A(n_S, n_D) + alpha(n_S, n_D)**2) * par for n_S in n]



def std_fish_biased2(n, n_D, par):
    """Error on variance from Fisher matrix with biased inverse covariance estimate,
       with correction, IK17 (26), ignoring 2A in denominator.
    """

    return [np.sqrt(2 * A(n_S, n_D) * ((n_S - n_D - 1))) / (alpha(n_S, n_D)**2) * par for n_S in n]



def std_fish_biased_TJK13(n, n_D, par):
    """0th-order error on variance from Fisher matrix with biased inverse covariance estimate.
       From TJK13 (49) with A (27) instead of A_corr (28) in (49)
    """

    #return std_fish_deb(n, n_D, par) / alpha(n, n_D)  # checked

    return [np.sqrt(2 * A(n_S, n_D) / alpha(n_S, n_D)**4 * (n_S - n_D - 1)) * par for n_S in n]



def std_fish_deb_TJ14(n, n_D, par):
    """Improved error on variance from the Fisher matrix.
       From TJ14 (12). This seems to be the case of the debiased precision matrix..
    """

    n_P = 2  # Number of parameters
    return [np.sqrt(2 * (n_S - n_D + n_P - 1) / (n_S - n_D -2)**2) * par for n_S in n]



def std_fish_biased_TJ14(n, n_D, par):
    """Improved error on variance from the Fisher matrix.
       From TJ14 (12), with division by the de-biasing factor alpha.
    """

    n_P = 2  # Number of parameters
    return [np.sqrt(2 * (n_S - n_D + n_P - 1) / (n_S - n_D -2)**2) / alpha(n_S, n_D) * par for n_S in n]



def hatdetF(n_S, n_D, sig2, delta):
    """Return expectation value of estimated determinant.
    """

    det = detF(n_D, sig2, delta)
    det = det * (2 * A(n_S, n_D) + alpha(n_S, n_D)**2)
    return det 



def deltaG2(a, n_S, n_D, sig2, delta):
    """Return <(Delta G_aa)^2>.
    """

    pref = 2 * A(n_S, n_D) / sig2**2 * (n_S - n_D - 1) * n_D**2

    if a==0:
        dG2 = pref
    elif a==1:
        dG2 = pref * (delta**2/12.0)**2
        # TODO: n_D**2 check
    else:
        error('Invalid parameter index {}'.format(a))

    return dG2



def std_fish_biased_ana(a, n, n_D, sig2, delta):

    return [np.sqrt(1.0/hatdetF(n_S, n_D, sig2, delta)**2 *
               deltaG2(a, n_S, n_D, sig2, delta) 
            ) for n_S in n]



def std_fish_biased_exa(a, n, n_D, sig2, delta):

    return [np.sqrt(1.0/detF(n_D, sig2, delta)**2 *
               deltaG2(a, n_S, n_D, sig2, delta) 
            ) for n_S in n]




def plot_det(n, x, sig2, delta, F, n_R):

    n_D = len(x)

    plot_init(n_D, n_R)
    ax = plt.subplot(1, 1, 1)

    det_num = []
    det_ana = []
    det_exa = np.array([detF(n_D, sig2, delta)] * len(n))

    for i, nn in enumerate(n):
        if n_R > 1:
            fmean = F[i,:].mean(axis=1)  # average over run
        else:
            fmean = F[i,0,:]  # check!
        det = fmean[0,0] * fmean[1,1] - fmean[0,1] ** 2
        det_num.append(det)

        det = hatdetF(nn, n_D, sig2, delta)
        det_ana.append(det)

    det_num = np.array(det_num)
    det_ana = np.array(det_ana)

    f = 1.005
    plt.plot(n/f, det_num/det_exa, marker='o', color='m', label='$|F|$ num/exact')
    plt.plot(n/f, det_ana/det_exa, 'y-', label='$|F|$ analytical/exact')
    #plt.plot(n/f, det_exa, 'c-', label='$|F|$ exa')

    det_exa_m1 = 1/det_exa
    det_num_m1 = 1/det_num
    det_ana_m1 = 1/det_ana

    plt.plot(n*f, det_num_m1/det_exa_m1, marker='s', color='m', linestyle='--', label='$1/|F|$ num/exact')
    plt.plot(n*f, det_ana_m1/det_exa_m1, 'y--', label='$1/|F|$ analytical/exact')

    plt.plot(n, det_exa/det_exa, 'b-')

    plt.xlabel('$n_{\\rm s}$')
    plt.ylabel('determinant of Fisher matrix relative to exact one')
    plt.legend(loc='best', numpoints=1, frameon=False)
    ax.set_yscale('log')
    plt.ylim(1e-4, 1e4)
    plt.savefig('det.pdf')

    f = open('det.txt', 'w')
    print >>f, '# n_S det_num det_ana det_exa'
    for i, nn in enumerate(n):
        print >>f, '{} {} {} {}'.format(nn, det_num[i], det_ana[i], det_exa[i])
    f.close()



def plot_std_fish_biased_ana(par_name, n, x, sig2, delta, F=None, n_R=0):

    n_D = len(x)
    plot_init(n_D, n_R)
    ax = plt.subplot(1, 1, 1)
    color = ['g', 'm']


    for i, p in enumerate(par_name):
        n_fine = np.arange(n[0], n[-1], len(n)/10.0)
        var_ana  = std_fish_biased_ana(i, n_fine, n_D, sig2, delta)
        plt.plot(n_fine, var_ana, '-', color=color[i], label='$\sigma[\sigma^2({})]$ '.format(p))

        var_exa  = std_fish_biased_exa(i, n_fine, n_D, sig2, delta)
        plt.plot(n_fine, var_exa, '--', color=color[i], label='$\sigma[\sigma^2({})]$ TKJ13'.format(p))

    mode = 3

    det = 0
        
    if F is not None:
        std_a = []
        std_b = []

        if mode == 4:
            fmean = F.mean(axis=(0,1))  # average over n_S and run
            det = fmean[0,0] * fmean[1,1] - fmean[0,1] ** 2

        for i, nn in enumerate(n):
            Fm1 = np.zeros(shape=(n_R, 2, 2))

            if mode == 3:
                if n_R > 1:
                    fmean = F[i,:].mean(axis=1)  # average over run
                else:
                    fmean = F[i,0,:]
                det = fmean[0,0] * fmean[1,1] - fmean[0,1] ** 2
                #print('{} {} {}'.format(nn, det, 1.0/det))
                # MKDEBUG 24/10: I can reproduce theory curve if I take theory det.
                # Uncomment following line.
                #det = 75000000.0
                # So maybe it is indeed higher-order terms that would reproduce data?
                # Or extra uncertainty from inverse in det?

            for run in range(n_R):
                f     = F[i,run]
                if mode == 1:    # Numerical inverse
                    Fm1[run] = np.linalg.inv(F[i,run])

                elif mode == 2:  # Analytical inverse
                    det = f[0,0] * f[1,1] - f[0,1] ** 2
                    Fm1[run,0,0] = f[1,1] / det
                    Fm1[run,1,1] = f[0,0] / det

                elif mode == 3:  # Analytical inverse with mean determinant (first-order term)
                    Fm1[run,0,0] = f[1,1] / det
                    Fm1[run,1,1] = f[0,0] / det

            std_a.append(np.std(Fm1[:,0,0]))
            std_b.append(np.std(Fm1[:,1,1]))
        plt.plot(n, std_a, marker='o', color='g', label='$\sigma[\sigma^2({})]$ from $F^{{-1}}$ (mode={})'.format('a', mode))
        plt.plot(n, std_b, marker='o', color='m', label='$\sigma[\sigma^2({})]$ from $F^{{-1}}$ (mode={})'.format('b', mode))

    plt.ylim(8e-9, 1e-2)
    plt.xlabel('$n_{\\rm s}$')
    plt.ylabel('std(var)')
    plt.legend(loc='best', numpoints=1, frameon=False)
    ax.set_yscale('log')
    plt.savefig('{}.pdf'.format('std_var_ana'))


def plot_A_alpha2(n, n_D, par):

    plot_init(n_D, -1)
    ax = plt.subplot(1, 1, 1)

    n_fine = np.arange(n[0], n[-1], len(n)/10.0)

    plt.plot(n_fine, 2*A(n_fine, n_D)/alpha(n_fine, n_D)**2, '-', color='g', label='$2A/\\alpha^2$')
    plt.legend(frameon=False)
    plt.xlabel('$n_{\\rm s}$')
    plt.ylabel('$2A/\\alpha^2$')
    ax.set_yscale('log')
    plt.savefig('A_alpha2')

    plot_init(n_D, -1)
    ax = plt.subplot(1, 1, 1)

    # It's strange that for low n_S the ratio 2A/alpha^2 approaches unity, but in std_fish_biased
    # ignoring 2A does not make any visible change

    pA = std_fish_biased(n_fine, n_D, par)
    pB = std_fish_biased2(n_fine, n_D, par)
    pC = std_fish_biased_TJ14(n_fine, n_D, par)
    plt.plot(n_fine, pA, '-', color='g', label='eq. (26)')
    plt.plot(n_fine, pB, '--', color='r', label='ignoring $2A$')
    plt.plot(n_fine, pC, '-.', color='r', label='TJK14')

    plt.legend(frameon=False)
    plt.xlabel('$n_{\\rm s}$')
    plt.ylabel('std(var)')
    ax.set_yscale('log')

    plt.savefig('std_var_comp')

    f = open('std_var_comp.txt', 'w')
    for i in range(len(n_fine)):
        print >>f, n_fine[i], pA[i], pB[i], pC[i]
    f.close()



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
        f_n_S_max = 10,
        spar = '1.0 0.0',
        sig2 = 5.0,
        mode   = 's',
        do_fit_stan = False,
        do_fish_ana = False,
        likelihood  = 'norm',
        n_jobs = 1,
        random_seed = False,
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
    parser.add_option('', '--f_n_S_max', dest='f_n_S_max', type='int', default=p_def.f_n_S_max,
        help='Maximum n_S = n_D x f_n_S_max, default: f_n_S_max={}'.format(p_def.f_n_S_max))

    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='list of parameter values, default=\'{}\''.format(p_def.spar))
    parser.add_option('-s', '--sig2', dest='sig2', type='float', default=p_def.sig2,
        help='variance of Gaussian, default=\'{}\''.format(p_def.sig2))

    parser.add_option('', '--fit_stan', dest='do_fit_stan', action='store_true',
        help='Run stan for MCMC fitting, default={}'.format(p_def.do_fit_stan))
    parser.add_option('', '--fish_ana', dest='do_fish_ana', action='store_true',
        help='Calculate analytical Fisher matrix, default={}'.format(p_def.do_fish_ana))
    parser.add_option('-L', '--like', dest='likelihood', type='string', default=p_def.likelihood,
        help='Likelihood for MCMC, one in \'norm\'|\'SH\', default=\'{}\''.format(p_def.likelihood))

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



def get_cov_ML_by_hand(y2, ymean, cov, n_S, n_D):
    """ Double check that get_cov_ML is the same as calculating 'by hand'"""

    cov_est = np.zeros(shape=(n_D, n_D))
    if n_D > 1:

        #didj = (y2 - ymean)*(y2 - ymean)
        #cov_est = didj.sum(axis=0)

        for i in range(n_D):
            for j in range(i, n_D):
                cov_est[i,j] = sum((y2[:,i] - ymean[i]) * (y2[:,j] - ymean[j]))
        for i in range(n_D):
            for j in range(0, i):
                cov_est[i,j] = cov_est[j,i]
        cov_est = cov_est / (n_S - 1.0)

    else:

        cov_est[0,0] = sum((y2 - ymean) * (y2 - ymean)) / (n_S - 1.0)

    return cov_est



def get_cov_ML(mean, cov, size):

    y2 = multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    # y2[:,j] = realisations for j-th data entry
    # y2[i,:] = data vector for i-th realisation

    # Estimate mean (ML)
    ymean = np.mean(y2, axis=0)

    # Calculate covariance matrix via np
    cov_est = np.cov(y2, rowvar=False)

    # Or by hand
    # 01/11 checked that this gives same results, with
    # test_ML.py -D 50 -p 1_0 --n_n_S 10 --f_n_S_max 10 -s 0.2 -m s -R 50 -v
    #n_D = len(mean)
    #cov_est = get_cov_ML_by_hand(y2, ymean, cov, size, n_D)
    #print(cov_est)


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
    plt.xlabel('$n_{\\rm s}$')
    plt.ylabel('normalised trace of ML covariance')

    ax = plt.subplot(1, 2, 2)
    plt.plot(n, sigma_m1_ML, 'b.')
    plt.plot([n[0], n[-1]], [1.0/sig2, 1.0/sig2], 'r-')
    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
    bias = [(n_S-1.0)/(n_S-n_D-2.0)/sig2 for n_S in n_fine]
    plt.plot(n_fine, bias, 'g-.')
    plt.xlabel('$n_{\\rm s}$')
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


def fit_corr_SH(x1, cov_true, cov_est_inv, n_jobs=3):
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



def simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options):
    """Simulate data"""
        
    if options.verbose == True:
        print('Creating {} simulations with {} runs each'.format(len(n_S_arr), options.n_R))

    cov = np.diag([options.sig2 for i in range(options.n_D)])            # *** cov of the data in the same catalog! ***

    # Go through number of simulations
    for i, n_S in enumerate(n_S_arr):

        if options.verbose == True:
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
            fish_num.F[i,run] = F

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
                    if options.verbose == True:
                        print('Running MCMC with mv normal likelihood')
                    res = fit_corr(x1, cov, cov_est, n_jobs=options.n_jobs)
                    fit = fit_norm
                elif options.likelihood == 'SH':
                    if options.verbose == True:
                        print('Running MCMC with Sellentin&Heavens (SH) likelihood')
                    res = fit_corr_SH(x1, cov, cov_est_inv, n_jobs=options.n_jobs)
                    fit = fit_SH
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




def write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options):
    """Write simulated runs to files"""

    if options.add_simulations == True:
        if options.verbose == True:
            print('Reading previous simulations from disk')

        # Initialise results
        n_n_S, n_R  = numbers_from_file(sigma_ML.file_base, 1)
        sigma_ML_prev    = Results(sigma_ML.par_name, n_n_S, n_R, file_base=sigma_ML.file_base)
        sigma_m1_ML_prev = Results(sigma_m1_ML.par_name, n_n_S, n_R, file_base=sigma_m1_ML.file_base)
        fish_ana_prev    = Results(fish_ana.par_name, n_n_S, n_R, file_base=fish_ana.file_base, fct=fish_ana.fct)
        fish_num_prev    = Results(fish_num.par_name, n_n_S, n_R, file_base=fish_num.file_base, fct=fish_num.fct, yscale='linear')
        fish_deb_prev    = Results(fish_deb.par_name, n_n_S, n_R, file_base=fish_deb.file_base, fct=fish_deb.fct)
        fit_norm_prev    = Results(fit_norm.par_name, n_n_S, n_R, file_base=fit_norm.file_base, fct=fit_norm.fct)
        fit_SH_prev      = Results(fit_SH.par_name, n_n_S, n_R, file_base=fit_SH.file_base, fct=fit_SH.fct)

        # Fill results from files
        read_from_file(sigma_ML_prev, sigma_m1_ML_prev, fish_ana_prev, fish_num_prev, fish_deb_prev, fit_norm_prev, \
                       fit_SH_prev, options)

        # Add new results
        sigma_ML.append(sigma_ML_prev)
        sigma_m1_ML.append(sigma_m1_ML_prev)
        fish_ana.append(fish_ana_prev)
        fish_num.append(fish_num_prev)
        fish_deb.append(fish_deb_prev)
        fit_norm.append(fit_norm_prev)
        fit_SH.append(fit_SH_prev)

    if options.verbose == True:
        print('Writing simulations to disk')

    if options.do_fish_ana == True:
        fish_ana.write_mean_std(n_S_arr)
    fish_num.write_mean_std(n_S_arr)
    fish_num.write_Fisher(n_S_arr)
    fish_deb.write_mean_std(n_S_arr)
    if options.do_fit_stan == True:
        if options.likelihood == 'norm':
            fit_norm.write_mean_std(n_S_arr)
        elif options.likelihood == 'SH':
            fit_SH.write_mean_std(n_S_arr)
        else:
             error('Invalid likelihood \'{}\''.format(options.likelihood))

    sigma_ML.write_mean_std(n_S_arr)
    sigma_m1_ML.write_mean_std(n_S_arr)



def read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options):
    """Read simulated runs from files"""

    if options.verbose == True:
        print('Reading simulations from disk')

    if options.do_fish_ana == True:
        fish_ana.read_mean_std()
    fish_num.read_mean_std()
    fish_num.read_Fisher()
    fish_deb.read_mean_std()

    if options.do_fit_stan:
        # Reading both irrespective of -L (likelihood) flag
        fit_norm.read_mean_std()
        fit_SH.read_mean_std()

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
    stop  = options.n_D * options.f_n_S_max
    n_S_arr = np.logspace(np.log10(start), np.log10(stop), options.n_n_S, dtype='int')
    n_n_S = len(n_S_arr)


    # Initialisation of results
    sigma_ML    = Results(tr_name, n_n_S, options.n_R, file_base='sigma_ML')
    sigma_m1_ML = Results(tr_name, n_n_S, options.n_R, file_base='sigma_m1_ML', yscale='log', fct={'mean': tr_N_m1_ML})

    fish_ana = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_ana', yscale='log', fct={'std': par_fish})
    fish_num = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_num', yscale='log', \
                       fct={'std': par_fish, 'std_var': std_fish_biased, 'std_var_TJK13': std_fish_biased_TJK13, 'std_var_TJ14': std_fish_biased_TJ14})
    fish_deb = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_deb', yscale='log', \
                       fct={'std': no_bias, 'std_var_TJK13': std_fish_deb, 'std_var_TJ14': std_fish_deb_TJ14})
    fit_norm = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_norm', yscale=['linear', 'log'],
                       fct={'std': par_fish, 'std_var': std_fish_biased})
    fit_SH   = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_SH', yscale=['linear', 'log'],
                       fct={'std': par_fish, 'std_var': std_fish_biased})

    # Data
    x1 = uniform.rvs(loc=-delta/2, scale=delta, size=options.n_D)        # exploratory variable
    x1.sort()
    yreal = options.a * x1 + options.b

    # Create simulations
    if re.search('s', options.mode) is not None:
 
        simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options) 

        write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options)


    # Read simulations
    if re.search('r', options.mode) is not None:

        read_from_file(sigma_ML, sigma_m1_ML, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options)


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
        fit_SH.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact})

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
        fit_SH.plot_std_var(n_S_arr, options.n_D, par=dpar2)

    sigma_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [options.sig2]})
    sigma_m1_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [1/options.sig2]})

    plot_std_fish_biased_ana(par_name, n_S_arr, x1, options.sig2, delta, F=fish_num.F, n_R=options.n_R)
    plot_det(n_S_arr, x1, options.sig2, delta, fish_num.F, options.n_R)
    plot_A_alpha2(n_S_arr, options.n_D, dpar2[1])

    ### End main program

    if options.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

