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
from scipy.stats import uniform, gamma
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


def model(p):
    """Linear model.

    input: p - dict: keywords 
                a, scalar - angular coefficient
                b, scalar - linear coefficient
                sig, scalar - scatter
                xmin, xmax, int - bounderies for explanatory variable
                nobs, int - number of observations in a catalog
                cov, matrix - covariance matrix between observations
                

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sig          
    """
    x = uniform.rvs(loc=p['xmin'], scale=p['xmax'] - p['xmin'], size=int(p['nobs']))
    
    x.sort()
    ytrue = np.array(p['a']*x + p['b'])

    y = [norm.rvs(loc=ytrue[i], scale=p['cov']) for i in range(len(x))]

    return np.array([[x[i], y[i]] for i in range(int(p['nobs']))])

def model_3par(p):
    """Linear model.

    input: p - dict: keywords 
                a, scalar - angular coefficient
                b, scalar - linear coefficient
                sig, scalar - scatter
                xmin, xmax, int - bounderies for explanatory variable
                nobs, int - number of observations in a catalog
                cov, matrix - covariance matrix between observations
                

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sig          
    """
    x = uniform.rvs(loc=p['xmin'], scale=p['xmax'] - p['xmin'], size=int(p['nobs']))
    
    x.sort()
    ytrue = np.array(p['a']*x + p['b'])

    y = [norm.rvs(loc=ytrue[i], scale=p['sig']) for i in range(len(x))]

    return np.array([[x[i], y[i]] for i in range(int(p['nobs']))])



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

def gamma_prior(par, func=False):
    """
    Gamma prior.
  
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
    dist = gamma(par['pgamma'])
    flag = False  
    while flag == False:   
        draw = dist.rvs() 
        if par['min'] < draw and draw < par['max']:
            flag = True
     
    if func == False:
        return draw
    else:
        return dist


def linear_dist(d2, p):
    """
    Distance between observed and simulated catalogues. 

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    data_sim = {}
    data_sim['x'] = d2[:,0]
    data_sim['y'] = d2[:,1]

    data_obs = {}
    data_obs['x'] = p['dataset1'][:,0]
    data_obs['y'] = p['dataset1'][:,1]

    formula = 'y ~ x + 1'

    mod_sim = smf.glm(formula=formula, data=data_sim, family=sm.families.Gaussian()).fit()
    mod_obs = smf.glm(formula=formula, data=data_obs, family=sm.families.Gaussian()).fit()

    res = np.sqrt(pow(mod_sim.params[0] - mod_obs.params[0], 2) + pow(mod_sim.params[1] - mod_obs.params[1], 2))

    return np.atleast_1d(res)


def linear_dist_scale(d2, p):
    """
    Distance between observed and simulated catalogues. 

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    data_sim = {}
    data_sim['x'] = d2[:,0]
    data_sim['y'] = d2[:,1]

    data_obs = {}
    data_obs['x'] = p['dataset1'][:,0]
    data_obs['y'] = p['dataset1'][:,1]

    formula = 'y ~ x + 1'

    mod_sim = smf.glm(formula=formula, data=data_sim, family=sm.families.Gaussian()).fit()
    mod_obs = smf.glm(formula=formula, data=data_obs, family=sm.families.Gaussian()).fit()

    res = np.sqrt(pow(mod_sim.params[0] - mod_obs.params[0], 2) + pow(mod_sim.params[1] - mod_obs.params[1], 2) + pow(mod_sim.scale - mod_obs.scale, 2))

    return np.atleast_1d(res)
    
