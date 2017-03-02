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
from scipy import interpolate

def model(x, p):
    """Linear model.

    input: p - dict: keywords 
                a, scalar - angular coefficient
                b, scalar - linear coefficient
                sig, scalar - scatter

          x, array

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sig          
    """
    y = np.array([norm.rvs(loc=p['a']*x[i] + p['b'], scale=p['sig']) for i in range(len(x))])

    return np.atleast_2d(y)


def sim_linear_ind(v):
    """
    Linear model simulator.
    Samples a normally distributed random variable 
    v['n'] times, having  mean =  v['a'] * v['xaxis'] + v['b'] and 
    variance = v['sigma']. 

    input: v -> dictionary of input parameters
           if v['xaxis'] is a file name, read data for x axis
           else simulate and store in 'xaxis.dat'

    output: array 
    """

    # check if observed xaxis exits, simulate if not
    if os.path.isfile(str(v['xaxis'][0])):
        # read xaxis values
        v['xaxis'] = np.readtxt(v['xaxis'][0])

        # simulate response variable
        y = v['a']* v['xaxis'] + v['b']

    else:
        # simulate xaxis values
        v['xaxis'] = np.random.uniform(v['xmin'], v['xmax'], size=int(v['nobs']))

        # simulate response variable
        y = model(v['xaxis'], v)

        # write xaxis data
        op1 = open('xaxis.dat', 'w')
        for i in range(int(v['nobs'])):
            op1.write(str(v['xaxis'][i]) + '\n')
        op1.close()

    dist = [norm(loc=y[i], scale=v['sig']) for i in range(len(y))]

    l1 = np.array([dist[i].rvs() for i in range(len(y))])


    return v['xaxis'], np.atleast_2d(l1).T

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


def linear_dist(d2, p):
    """
    Distance between observed and simulated catalogues. 

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    y_obs = p['dataset1'][1].flatten()
    y_sim = d2[1].flatten()
    x = p['dataset1'][0].flatten()

    print y_obs.shape
    print y_sim.shape
    print x.shape

    f_obs = interpolate.interp1d(x, y_obs, kind='cubic')
    f_sim = interpolate.interp1d(x, y_sim, kind='cubic')
 

    # test angular coefficient
    ang = sum([(y_obs[i] - y_sim[i]) ** 2 for i in range(len(y_sim))])
   
    # test linear coefficient
    lin = (f_obs(0) - f_sim(0)) ** 2

    # test scatter
    s = np.std(y_obs - f_sim(x)) + np.std(y_sim - f_obs(x))

    return s, lin, ang
    
  
    

    
