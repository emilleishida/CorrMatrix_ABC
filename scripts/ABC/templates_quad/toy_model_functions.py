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



def model_cov(p):
    """Linear model.

    input: p - dict: keywords 
                amp, scalar - amplitude coefficient
                tilt, scalar - tilt coefficient
                sig, scalar - scatter
                xmin, xmax, int - bounderies for explanatory variable
                nobs, int - number of observations in a catalog
                cov, matrix - covariance matrix between observations
                

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sig          
    """

    try:                
        logell = p['dataset1'][:,0]
    except KeyError:
        raise ValueError('Observed data not found in model')

    logell.sort()

    # Set (amp, tilt, a), with quadratic coefficient a being constant

    param = p['amp'], p['tilt'], -0.19338417)
    q     = quadratic(logell, *param)
    ytrue = 10**q

    if isinstance(p['cov'], float):
        raise ValueError('Covariance is not a matrix!')

    y = multivariate_normal.rvs(mean=ytrue, cov=p['cov'])

    return np.array([[logell[i], y[i]] for i in range(int(p['nobs']))])



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
       least squres between observed and simulated data points y.

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

    if bool(p['xfix']) == False:
        raise ValueError('Parameter xfix needs to be 1 for linear_dist_data distance')

    y_sim = d2[:,1]
    y_obs = p['dataset1'][:,1]

    y_delta = y_sim - y_obs
    
    # Unweighted distances
    #dist    = np.sqrt(sum(y_delta**2))

    # Least squares weighted by covariance
    cov_est_inv = p['cov_est_inv']
    dist = np.einsum('i,ij,j', y_delta, cov_est_inv, y_delta)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)

