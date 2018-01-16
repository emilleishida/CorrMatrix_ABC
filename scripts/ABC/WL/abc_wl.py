#!/usr/bin/env python

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from wl_functions import linear_dist, model_Cl, gaussian_prior, model_cov

import numpy as np
import os
import re

from scipy.stats import uniform, multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW

from astropy import units

import nicaea_ABC
import scipy.stats._multivariate as mv



def get_cov_Gauss(ell, C_ell, f_sky, sigma_eps, nbar):
    """Return Gaussian covariance.
    
    Parameters
    ----------
    ell: array of double
         angular Fourier modes
    C_ell: array of double
         power spectrum
    f_sky: double
        sky coverage fraction
    sigma_eps: double
        ellipticity dispersion (per component)
    nbar: double
        galaxy number density [rad^{-2}]

    Returns
    -------
    Sigma: matrix of double
        covariance matrix
    """

    # Total (signal + shot noise) power spectrum
    C_ell_tot = C_ell + sigma_eps**2 / (2 * nbar)

    D         = 1.0 / (f_sky * (2.0 * ell + 1)) * C_ell_tot**2
    Sigma = np.diag(D)

    return Sigma


def sample_cov_Wishart(cov, n_S):
    """Returns estimated coariance as sample from Wishart distribution
 
    Parameters
    ----------
    cov: matrix of double
         'true' covariance matrix (scale matrix)
    n_S: int
         number of simulations, dof = nu = n_S - 1

    Returns
    -------
    cov_est: matrix of double
         sampled matrix
    """

    # Sample covariance from Wishart distribution, with dof nu=n_S - 1
    W = mv.wishart(df=n_S - 1, scale=cov)
    cov_est = W.rvs() / (n_S - 1)  ## Divide or not??

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


def weighted_std(data, weights): 
    """Taken from http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf"""

    mean = np.average(data, weights=weights)
    c = sum([weights[i] > pow(10, -6) for i in range(weights.shape[0])])

    num = sum([weights[i] * pow(data[i] - mean, 2) for i in range(data.shape[0])])
    denom = (c - 1) * sum(weights)/float(c)

    return np.sqrt(num / denom)

#user input file
filename = 'wl_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['Omega_m'] = float(Parameters['Omega_m'][0])
Parameters['sigma_8'] = float(Parameters['sigma_8'][0])

Parameters['f_sky'] = float(Parameters['f_sky'][0])
Parameters['sigma_eps'] = float(Parameters['sigma_eps'][0])
Parameters['nbar'] = float(Parameters['nbar'][0])

Parameters['lmin'] = float(Parameters['lmin'][0])
Parameters['lmax'] = float(Parameters['lmax'][0])
Parameters['nell'] = int(Parameters['nell'][0])
Parameters['simulation_input']['lmin'] = Parameters['lmin']
Parameters['simulation_input']['lmax'] = Parameters['lmax']
Parameters['simulation_input']['nell'] = Parameters['nell']

Parameters['path_to_nicaea'] = Parameters['path_to_nicaea'][0]
#if Parameters['path_to_nicaea'][0] != '/':
    # Relative path, add $HOME
    #Parameters['path_to_nicaea'] = '{}/{}'.format(os.environ['HOME'], Parameters['path_to_nicaea'])

# Replace $VAR with environment variable value
Parameters['path_to_nicaea'] = re.sub('(\$\w*)', os.environ['NICAEA'], Parameters['path_to_nicaea'])

Parameters['simulation_input']['path_to_nicaea'] = Parameters['path_to_nicaea']

# set functions
Parameters['simulation_func'] = model_Cl
#Parameters['simulation_func'] = model_cov
Parameters['distance_func'] = linear_dist
Parameters['prior']['Omega_m']['func'] = gaussian_prior
Parameters['prior']['sigma_8']['func'] = gaussian_prior


# Get 'observation' (fiducial model)
# Call nicaea
nicaea_ABC.run_nicaea(Parameters['path_to_nicaea'], Parameters['lmin'], \
    Parameters['lmax'], Parameters['nell'])

# Read nicaea output
ell, C_ell_obs = nicaea_ABC.read_Cl('.', 'P_kappa')


# add to parameter dictionary
Parameters['dataset1'] = np.array([[ell[i], C_ell_obs[i]] for i in range(Parameters['nell'])])

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

#############################################
### Covariance

# Construct (true) covariance Sigma
nbar_amin2  = units.Unit('{}/arcmin**2'.format(Parameters['nbar']))
nbar_rad2   = nbar_amin2.to('1/rad**2')
# We use the same C_ell as the 'observation', from above
cov         = get_cov_Gauss(ell, C_ell_obs, Parameters['f_sky'], Parameters['sigma_eps'], nbar_rad2)

# Estimate covariance as sample from Wishart distribution
Parameters['nsim'] = int(Parameters['nsim'][0])
cov_est = sample_cov_Wishart(cov, Parameters['nsim'])
#cov_est = get_cov_ML(ytrue, cov, Parameters['nsim'])

# add covariance to user input parameters
Parameters['simulation_input']['cov'] = cov_est
#############################################

#initiate ABC sampler
sampler_ABC = ABC(params=Parameters)

#build first particle system
sys1 = sampler_ABC.BuildFirstPSystem()

#update particle system until convergence
sampler_ABC.fullABC()


# calculate numerical results
op1 = open(Parameters['file_root'] + str(sampler_ABC.T) + '.dat', 'r')
lin1 = op1.readlines()
op1.close()

data1 = [elem.split() for elem in lin1]

a_samples = np.array([float(line[0]) for line in data1[1:]])
b_samples = np.array([float(line[1]) for line in data1[1:]])

weights = np.loadtxt(Parameters['file_root'] + str(sampler_ABC.T) + 'weights.dat')

a_results = DescrStatsW(a_samples, weights=weights, ddof=0)
b_results = DescrStatsW(b_samples, weights=weights, ddof=0)

a_results.std_mean = weighted_std(a_samples, weights)
b_results.std_mean = weighted_std(b_samples, weights)

# store numerical results
op2 = open('num_res.dat', 'w')
op2.write('a_mean    ' + str(a_results.mean) + '\n')
op2.write('a_std     ' + str(a_results.std_mean) + '\n\n\n')
op2.write('b_mean    ' + str(b_results.mean) + '\n')
op2.write('b_std     ' + str(b_results.std_mean))
op2.close()

print 'Numerical results:'
print 'a:    ' + str(a_results.mean) + ' +- ' + str(a_results.std_mean)
print 'b:    ' + str(b_results.mean) + ' +- ' + str(b_results.std_mean)



#plot results
plot_2p( sampler_ABC.T, 'results.pdf' , Parameters)
