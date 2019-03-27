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
from scipy.stats import norm,  multivariate_normal
from scipy.stats import uniform
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from covest import model_quad



def model_cov(p):
    """Linear model.

    input: p - dict: keywords 
                ampl, scalar - amplitude coefficient
                tilt, scalar - tilt coefficient
                sig, scalar - scatter
                xmin, xmax, int - bounderies for explanatory variable
                cov, matrix - covariance matrix between observations
                

    output: [x, y], array - draw from normal distribution using cov matrix
    """

    if 'dataset1' in p:
         # Get abscissa values from dataset in parameter
         x = p['dataset1'][:,0]
    else:
        fname = 'dataset1.txt'
        if os.path.isfile(fname):
            print('Loading dataset1 from file \'dataset1.txt\'')
            dat = np.loadtxt(fname)
            x   = dat[:,0]
        else:
            print('Observed data not found in model, neither in file nor in function arguments')
            sys.exit(1)


    # Get q quadratic function in u = logell

    # If x = logell
    u = x

    # If x = ell
    # u = np.log10(x)

    # Ordinate
    y_true = model_quad(u, p['ampl'], p['tilt'])

    if not 'cov' in p:
        print('Loading estimated covariance from file \'cov_est.txt\'')
        cov_est = np.loadtxt('cov_est.txt')
    else:
        if isinstance(p['cov'], float):
            raise ValueError('Covariance is not a matrix!')
        else:
            cov_est = p['cov']

    # Model
    y = multivariate_normal.rvs(mean=y_true, cov=cov_est)

    nx = len(x)

    simulation = np.array([[x[i], y[i]] for i in range(nx)])
    #print('MKDEBUG saving model with params={} {}'.format(p['ampl'], p['tilt']))
    #np.savetxt('simulation_{}_{}.txt'.format(p['ampl'], p['tilt']), simulation)

    return simulation



def gaussian_prior(par, func=False):
    """
    Gaussian prior.
  
    input: par -> dictionary of parameter values
                  keywords: mean, standard_devitation, 
                            min and max
                  values: all scalars 
           func -> boolean (optional)
                   if True returns the pdf random variable. 
                   Default is False.
    output: scalar (if func=False)
            gaussian probability distribution function (if func=True)
    """

    np.random.seed()    
    dist = norm(loc=par['pmean'], scale=par['pstd'])
    flag = False  
    while flag == False:   
        draw = dist.rvs() 
        if par['min'] < draw and draw < par['max']:
            flag = True
     
    if func == False:
        return draw
    else:
        return dist



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
        print('linear_dist_data: Reading cov_true_inv.txt from disk')
        cov_inv = np.loadtxt('cov_true_inv.txt')

    dist = np.einsum('i,ij,j', dC, cov_inv, dC)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)



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



# See WL/w_functions.py


def acf_one(C, di, mean, reverse=False):
    """Return one value of the auto-correlation function xi(x) of C at argument x=di

    Parameters
    ----------
    C: array(float)
        observed power spectrum
    di: int
        difference of ell-mode indices
    mean: float
        mean value of C (can be 0 if un-centered acf is desired)
    reverse: bool, optional, default=False
        if True, reverse one of the vectors

    Returns
    -------
    xi: float
        auto-correlation function value
    """

    n  = len(C)
    # Shift signal and keep to same length (loose entries at high-ell end)
    C1 = C[:n-di]
    C2 = C[di:]

    if reverse:
        C2 = C2[::-1]

    # Estimate ACF
    xi = sum((C1 - mean) * (C2 - mean)) / float(n)

    return xi



def acf(C, norm=False, centered=False, reverse=False):
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
    xi: array of float
        auto-correlation function value
    """

    if centered:
        mean = 0
    else:
        mean = C.mean()

    xi = []
    for di in range(len(C)):
        xi.append(acf_one(C, di, mean, reverse=reverse))

    if norm:
        # Var = < C_ell C_ell> = xi(0)
        xi = xi / xi[0]

    return xi



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

    if 'cov' in p:
        cov = p['cov']
    else:
        cov = np.loadtxt('cov_est.txt')

    # Weighted data points
    C_ell_sim_w = C_ell_sim / np.sqrt(np.diag(cov))
    C_ell_obs_w = C_ell_obs / np.sqrt(np.diag(cov))

    xi = acf(C_ell_obs, norm=True, centered=False, reverse=False)

    d = 0
    n = len(C_ell_obs)
    for i in range(n):
        for j in range(n):
            term = (C_ell_sim_w[i] - C_ell_obs_w[j])**2 * xi[np.abs(i-j)]**2
            d = d + term

    d = np.atleast_1d(d)
    return d

