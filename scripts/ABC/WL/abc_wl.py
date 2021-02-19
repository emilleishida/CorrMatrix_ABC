#!/usr/bin/env python

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from wl_functions import linear_dist_data, linear_dist_data_diag, linear_dist_data_SVD, model_Cl_norm, linear_dist_data_acf, acf

import numpy as np
import os
import re
import sys

from scipy.stats import multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW

import nicaea_ABC

from covest import get_cov_ML, get_cov_Gauss, weighted_std, get_cov_WL



#user input file
filename = 'wl_model.input'


#read  user input
Parameters = read_input(filename)

# Possible input parameters.
pars = ['Omega_m', 'sigma_8']

for par in pars:
    if par in Parameters:
        Parameters[par] = float(Parameters[par][0])

Parameters['f_sky'] = float(Parameters['f_sky'][0])
Parameters['sigma_eps'] = float(Parameters['sigma_eps'][0])
Parameters['nbar'] = float(Parameters['nbar'][0])

Parameters['lmin'] = float(Parameters['lmin'][0])
Parameters['lmax'] = float(Parameters['lmax'][0])
Parameters['nell'] = int(Parameters['nell'][0])
Parameters['simulation_input']['lmin'] = Parameters['lmin']
Parameters['simulation_input']['lmax'] = Parameters['lmax']
Parameters['simulation_input']['nell'] = Parameters['nell']

#Parameters['path_to_nicaea'] = Parameters['path_to_nicaea'][0]
#if Parameters['simulation_input']['path_to_nicaea'][0] != '/':
    # Relative path, add $HOME
    #Parameters['simulation_input']['path_to_nicaea'] = '{}/{}'.format(os.environ['HOME'], Parameters['simulation_input']['path_to_nicaea'])

# Replace $VAR with environment variable value
#Parameters['path_to_nicaea'] = re.sub('(\$\w*)', os.environ['NICAEA'], Parameters['path_to_nicaea'])

### Dictionaries for functions

# Simulation model
simulation = {'model_Cl_norm': model_Cl_norm}

# Distance
distance = {'linear_dist_data_diag': linear_dist_data_diag,
            'linear_dist_data':      linear_dist_data,
            'linear_dist_data_SVD':  linear_dist_data_SVD,
            'linear_dist_data_acf':  linear_dist_data_acf}

# set functions
Parameters['simulation_func'] = simulation[Parameters['simulation_func'][0]]
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

# Priors are set in read_input

# Get 'observation' (fiducial model)
# Call nicaea
nicaea_ABC.run_nicaea(Parameters['lmin'], Parameters['lmax'], Parameters['nell'])

# Read nicaea output
ell, C_ell_obs = nicaea_ABC.read_Cl('.', 'P_kappa')


# add to parameter dictionary
Parameters['dataset1'] = np.array([[ell[i], C_ell_obs[i]] for i in range(Parameters['nell'])])

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

### Test of acf
xi_u = acf(Parameters['dataset1'][:,1], norm=False, centered=False)
xi = acf(Parameters['dataset1'][:,1], norm=True, centered=False)
xi_c = acf(Parameters['dataset1'][:,1], norm=True, centered=True)
print('# i xi_uu xi_n xi_nc')
for di in range(Parameters['nell']):
    print('{} {} {} {}'.format(di, xi_u[di], xi[di], xi_c[di]))

sys.exit(0)

#############################################
### Covariance

Parameters['nsim'] = int(Parameters['nsim'][0])
cov, cov_est = get_cov_WL('Gauss', ell, C_ell_obs, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'])


# add covariance to user input parameters, to be used in model
Parameters['cov'] = cov_est
Parameters['simulation_input']['cov'] = cov_est

if distance_str == 'linear_dist_data':
    cov_true_inv = np.linalg.inv(cov)
    Parameters['cov_true_inv'] = cov_true_inv
    np.savetxt('cov_true_inv.txt', cov_true_inv)

elif distance_str == 'linear_dist_data_SVD':
    # SVD
    u, s, vh = np.linalg.svd(cov_est, full_matrices=True)

    sm   = 1.0 / s
    nsim = Parameters['nsim']
    # Set inverse diagonal of singular elements to zero 
    sm[nsim:] = 0

    # Pseudo-inverse
    cov_inv_P = np.dot(vh.conj().T * sm, u.conj().T)
    Parameters['cov_est_inv_P'] = cov_inv_P
    np.savetxt('cov_est_inv_P.txt', cov_inv_P)

    # Test: Compare pseudo-inverse to inverse in case of non-singular matrix
    #print('cov_inv_P', cov_inv_P[0,0], cov_inv_P[0,1])
    #cov_inv = np.linalg.inv(cov_est)
    #print('cov_inv', cov_inv[0,0], cov_inv[0,1])


# cov_est.txt on disk is read when running plot/continue/test_ABC.py.
np.savetxt('cov_est.txt', cov_est)

print('Starting ABC sampling...')

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
print ''


try:
    #plot results
    plot_2p( sampler_ABC.T, 'results.pdf' , Parameters)
except:
    print('Plotting ABC to results.pdf failed, maybe display not available.')

