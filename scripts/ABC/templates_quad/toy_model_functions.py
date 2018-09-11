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



def add(amp):
    """Return the additive constant of the quadratic function
       from the amplitude fitting parameter
       (mimics power-spectrum normalisation s8)
    """
    
    # This provides a best-fit amp=0.827, but the 10% increased
    # spectrum (0.9097) gives a best-fit of 0.925
    # Changing the prefactor of amp or lg(amp) does not help...
    c = np.log10(amp)*2 - 6.11568527 + 0.1649
    
    return c



def shift(tilt):
    """Return the shift parameter of the quadratic function
       from the tilt parameter (mimics matter density)
    """
    
    x0 = tilt * 1.85132114 / 0.306

    return x0



def quadratic(x, *params):
    """Used to fit quadratic function varying all three parameters
    """
    
    (amp, tilt, a) = np.array(params)
    c  = add(amp)
    x0 = shift(tilt)
    
    return c + a * (x - x0)**2



def quadratic_amp_tilt(x, amp, tilt):
    """Return quadratic function given coordinate 1 (=x=logell), amplitude,
       and tilt.
    """

    param   = (amp, tilt, -0.17586216)
    q       = quadratic(x, *param)

    return 10**q



def model_cov(p):
    """Linear model.

    input: p - dict: keywords 
                amp, scalar - amplitude coefficient
                tilt, scalar - tilt coefficient
                sig, scalar - scatter
                xmin, xmax, int - bounderies for explanatory variable
                cov, matrix - covariance matrix between observations
                

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sig          
    """

    try:                
        logell = p['dataset1'][:,0]
    except KeyError:
        raise ValueError('Observed data not found in model')


    Cell_true = quadratic_amp_tilt(logell, p['amp'], p['tilt'])

    if isinstance(p['cov'], float):
        raise ValueError('Covariance is not a matrix!')

    Cell = multivariate_normal.rvs(mean=Cell_true, cov=p['cov'])

    nell = len(logell)

    return np.array([[logell[i], Cell[i]] for i in range(nell)])



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
        #print('linear_dist_data: Reading cov_true_inv.txt from disk')
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

