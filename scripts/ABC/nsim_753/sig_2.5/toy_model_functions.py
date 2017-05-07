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
 
    y = multivariate_normal.rvs(mean=ytrue, cov=p['cov'], size=1)

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


def linear_dist(d2, p):
    """
    Distance between observed and simulated catalogues. 

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    y_obs = p['dataset1'][:,1]
    x = p['dataset1'][:,0]

    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(d2[:,0],d2[:,1])
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x,y_obs)

    ang = np.sqrt((slope1 - slope2) ** 2  + (intercept1 - intercept2) ** 2)
   

    return np.atleast_1d(ang)
    
  
    

"""
def sim_linear_ind(v):
    ""
    Linear model simulator.
    Samples a normally distributed random variable 
    v['n'] times, having  mean =  v['a'] * v['xaxis'] + v['b'] and 
    variance = v['sigma']. 

    input: v -> dictionary of input parameters
           if v['xaxis'] is a file name, read data for x axis
           else simulate and store in 'xaxis.dat'

    output: array 
    ""

    # check if observed xaxis exits, simulate if not
    if os.path.isfile(str(v['xaxis'][0])):
        # read xaxis values
        v['xaxis'] = np.readtxt(v['xaxis'][0])

        # simulate response variable
        y = np.array([model(v['xaxis'][i], v) for i in range(len(v['xaxis']))])

    else:
        # simulate xaxis values
        v['xaxis'] = np.random.uniform(v['xmin'], v['xmax'], size=int(v['nobs']))

        # simulate response variable
        y = np.array([model(v['xaxis'][i], v) for i in range(len(v['xaxis']))])

        # write xaxis data
        op1 = open('xaxis.dat', 'w')
        for i in range(int(v['nobs'])):
            op1.write(str(v['xaxis'][i]) + '\n')
        op1.close()


    return v['xaxis'], np.atleast_2d(y).T
"""

    
