"""
Created by Emille Ishida in 10 June 2016.

Example of functions to be used as input to CosmoABC. 
You are free to customize this functions to your own problem
as long as you respect the input/ouput requirements and 
***
    update the function names into the keywords 

    distance_func
    simulation_func
    prior_func
  
    in the user input file
***. 

"""


import numpy as np
import os
import re
from scipy.stats import norm,  multivariate_normal, gamma
from scipy.stats import uniform
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

from CorrMatrix_ABC import nicaea_ABC



def model_Cl(p):
    """Return model angular power spectrum and read/get covariance.
    """


    # Assign parameter values.

    # Possible parameters.
    pars = ['Omega_m', 'sigma_8']
    par_val  = []
    par_name = []

    # Check which parameter is in input parameter list.
    for par in pars:
        if par in p:
            par_val.append(p[par])
            par_name.append(par) 

    # This is necessary for plot_ABC.py, which has not already performed the substitution in abc_wl.py
    #p['path_to_nicaea'] = re.sub('(\$\w*)', os.environ['NICAEA'], p['path_to_nicaea'])

    # Run nicaea to produce model Cl
    err, C_ell_name = nicaea_ABC.run_nicaea(p['lmin'], p['lmax'], p['nell'], \
                                par_name = par_name, par_val = par_val)

    if err != 0:

        ell       = np.zeros(shape=p['nell'])
        C_ell_est = np.zeros(shape=p['nell']) 

    else:

        ell, C_ell = nicaea_ABC.read_Cl('.', C_ell_name)
        os.unlink(C_ell_name)

        # Covariance assumed to be constant
        if 'cov' in p:
            # This script is called after abc_wl.py (ABC run)
            cov_est = p['cov']
        else:
            # This script is called from plot_ABC.py or test_ABC_distance.py: Need to get cov from disk
            #print('Reading cov_est.txt from disk')
            cov_est = np.loadtxt('cov_est.txt')

    return ell, C_ell, cov_est



def model_Cl_norm(p):
    """Return sample of model angular power spectrum from mv Gaussian.
    """

    ell, C_ell, cov_est = model_Cl(p)

    # Sample hat Cl from Norm(Cl, hat Sigma). Here, cov_est can be singular.
    C_ell_est = multivariate_normal.rvs(mean=C_ell, cov=cov_est)

    return np.array([[ell[i], C_ell_est[i]] for i in range(int(p['nell']))])



def model_Cl_gamma(p):
    """Return sample of model angular power spectrum from mv Gaussian.
    """


    ell, C_ell, cov_est = model_Cl(p)

    C_ell_est = np.zeros(shape = len(ell))
    for i, l in enumerate(ell):

        # dof
        nu = 2 * l + 1
    
        # shape parameter
        a  = nu/2

        # scale
        s = 2 / nu / C_ell[i]

    # Sample hat Cl from Norm(Cl, hat Sigma). Here, cov_est can be singular.
    C_ell_est = multivariate_normal.rvs(mean=C_ell, cov=cov_est)

    return np.array([[ell[i], C_ell_est[i]] for i in range(int(p['nell']))])


def acf_one(C, di, mean):
    """Return one value of the auto-correlation function xi(x) of C at argument x=di

    Parameters
    ----------
    C: array(float)
        observed power spectrum
    di: int
        difference of ell-mode indices
    mean: float
        mean value of C (can be 0 if un-centered acf is desired)

    Returns
    -------
    xi: float
        auto-correlation function value
    """

    n  = len(C)
    # Shift signal and keep to same length (loose entries at high-ell end)
    C1 = C[:n-di]
    C2 = C[di:]

    # Estimate ACF
    xi = sum((C1 - mean) * (C2 - mean)) / float(n)

    return xi


def acf(C, norm=False, centered=False):
    """Return auto-correlation function of C.

    Parameters
    ----------
    C: array(float)
        observed power spectrum
    di: int
        difference of ell-mode indices
    norm: bool, optional, default=False
        if True, acf is normalised by the variance
    centered: bool, optional, default=False
        if True, center acf by subtracting the mean

    Returns
    -------
    xi: float
        auto-correlation function value
    """

    if centered:
        mean = 0
    else:
        mean = C.mean()

    xi = []
    for di in range(len(C)):
        xi.append(acf_one(C, di, mean))

    if norm:
        # Var = < C_ell C_ell> = xi(0)
        xi = xi / xi[0]

    return xi 



def linear_dist_data_diag(d2, p):
    """Distance between observed and simulated catalogues using
       one over estimated diagonal covariance elements.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]

    dC = C_ell_sim - C_ell_obs

    if 'cov' in p:
        cov = p['cov']
    else:
        #print('linear_dist_data_diag: Reading cov_est.txt from disk')
        cov = np.loadtxt('cov_est.txt')

    # Least squares distance weighted by inverse diagonal elements of covariance
    dist = np.sqrt(sum(dC**2/np.diag(cov)))

    return np.atleast_1d(dist)



def linear_dist_data(d2, p):
    """Distance between observed and simulated catalogues using
       true inverse covariance.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]

    dC = C_ell_sim - C_ell_obs

    # Unweighted distances
    #dist    = np.sqrt(sum(dC**2))

    # Least squares weighted by covariance
    if 'cov_true_inv' in p:
        #print('linear_dist_data: Using true inverse covariance matrix')
        cov_inv = p['cov_true_inv']
    else:
        #print('linear_dist_data: Reading cov_true_inv.txt from disk')
        cov_inv = np.loadtxt('cov_true_inv.txt')

    dist = np.einsum('i,ij,j', dC, cov_inv, dC)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)



def linear_dist_data_SVD(d2, p):
    """Distance between observed and simulated catalogues using
       SVD of estimated inverse covariance.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]

    dC = C_ell_sim - C_ell_obs

    # Pseudo-inverse of estimated covariance (from SVD)
    if 'cov_est_inv_P' in p:
        cov_est_inv_P = p['cov_est_inv_P']
    else:
        cov_est_inv_P = np.loadtxt('cov_est_inv_P.txt')

    dist = np.einsum('i,ij,j', dC, cov_est_inv_P, dC)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)


def linear_dist_data_acf(d2, p):
    """Distance between observed and simulated catalogues using
       the auto-correlation function of the observation

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]


