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
from scipy.stats import norm,  multivariate_normal
from scipy.stats import uniform
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

import nicaea_ABC



def model_Cl(p):
    """Return model angular power spectrum.
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
    p['path_to_nicaea'] = re.sub('(\$\w*)', os.environ['NICAEA'], p['path_to_nicaea'])

    # Run nicaea to produce model Cl
    err, C_ell_name = nicaea_ABC.run_nicaea(p['path_to_nicaea'], p['lmin'], p['lmax'], p['nell'], \
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
            print('Reading cov_est.txt from disk')
            cov_est = np.loadtxt('cov_est.txt')

        # Sample hat Cl from Norm(Cl, hat Sigma)
        C_ell_est = multivariate_normal.rvs(mean=C_ell, cov=cov_est)

    return np.array([[ell[i], C_ell_est[i]] for i in range(int(p['nell']))])



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
        print('Reading cov_est.txt from disk')
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
        cov_inv = p['cov_true_inv']
    else:
        print('Reading cov_true_inv.txt from disk')
        cov_inv = np.loadtxt('cov_true_inv.txt')

    dist = np.einsum('i,ij,j', dC, cov_inv, dC)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)
