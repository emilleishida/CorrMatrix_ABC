#!/usr/bin/env python3

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from wl_functions import linear_dist_data, linear_dist_data_diag, model_Cl_norm

import numpy as np
import os
import re
import sys

from scipy.stats import multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW

from CorrMatrix_ABC import nicaea_ABC

from CorrMatrix_ABC.covest import get_cov_ML, get_cov_Gauss, weighted_std, get_cov_WL, linear_dist_data_acf2_lin, acf, get_ell_mode


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

Parameters['logellmin'] = float(Parameters['logellmin'][0])
Parameters['logellmax'] = float(Parameters['logellmax'][0])
Parameters['nell'] = int(Parameters['nell'][0])
Parameters['ellmode'] = Parameters['ellmode'][0]
Parameters['simulation_input']['logellmin'] = Parameters['logellmin']
Parameters['simulation_input']['logellmax'] = Parameters['logellmax']
Parameters['simulation_input']['nell'] = Parameters['nell']

### Dictionaries for functions

# Simulation model
simulation = {'model_Cl_norm': model_Cl_norm}

# Distance
distance = {'linear_dist_data_diag': linear_dist_data_diag,
            'linear_dist_data':      linear_dist_data,
            'linear_dist_data_acf2_lin': linear_dist_data_acf2_lin}

# set functions
Parameters['simulation_func'] = simulation[Parameters['simulation_func'][0]]
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

# Priors are set in read_input

# Get 'observation' (fiducial model)
# Call nicaea
pars = ['Omega_m', 'sigma_8']
par_val  = []
par_sval = []
par_name = []

# Check which parameter is in input parameter list.
for par in pars:
    if par in Parameters:
        par_name.append(par)
        par_val.append(Parameters[par])
nicaea_ABC.run_nicaea(10**Parameters['logellmin'], 10**Parameters['logellmax'],
                      Parameters['nell'],
                      par_name=par_name, par_val=par_val,
                      verbose=True)

# Read nicaea output'
pkappa_name_list = ['P_kappa']
for v in par_val:
    pkappa_name_list.append(str(v))
pkappa_name = '_'.join(pkappa_name_list)
ell, C_ell_obs = nicaea_ABC.read_Cl('.', pkappa_name)
logell = np.log10(ell)

# Check linear binning
if get_ell_mode(ell) != 'lin':
    raise ValueError('ell bins not linear')

# add to parameter dictionary
Parameters['dataset1'] = np.array([[logell[i], C_ell_obs[i]] for i in range(Parameters['nell'])])

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']
np.savetxt('dataset1.txt', Parameters['dataset1'], header='# log(ell) C_ell')


### Covariance

Parameters['nsim'] = int(Parameters['nsim'][0])
cov_model = Parameters['cov_model'][0]
# d_SSC=0.55 is 23% increase of total cov diag
#cov, cov_est = get_cov_WL(cov_model, ell, C_ell_obs, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'], d_SSC=0.55)
#cov, cov_est = get_cov_WL(cov_model, ell, C_ell_obs, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'], d_SSC=0.34)
cov, cov_est = get_cov_WL(cov_model, ell, C_ell_obs, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'], d_SSC=0.0)

# add covariance to user input parameters, to be used in model
Parameters['cov'] = cov_est
Parameters['simulation_input']['cov'] = cov_est

if distance_str == 'linear_dist_data':
    cov_true_inv = np.linalg.inv(cov)
    Parameters['cov_true_inv'] = cov_true_inv
    np.savetxt('cov_true_inv.txt', cov_true_inv)
elif distance_str == 'linear_dist_data_acf2_lin':
    mean_std_t = True
    y_input = Parameters['dataset1'][:,1]
    xi = acf(y_input, norm=True, count_zeros=False, mean_std_t=mean_std_t)
    Parameters['xi'] = xi

    # write to disk
    fout = open('xi.txt', 'w')
    for i, x in enumerate(xi):
        print('{} {}'.format(i, x), file=fout)
    fout.close()



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

# For some reason a_results.std_mean is NaN, and a_results_std is
# slightly different from the result of the following computation
a_std = weighted_std(a_samples, weights)
b_std = weighted_std(b_samples, weights)

# store numerical results
op2 = open('num_res.dat', 'w')
op2.write('a_mean    ' + str(a_results.mean) + '\n')
op2.write('a_std     ' + str(a_std) + '\n\n\n')
op2.write('b_mean    ' + str(b_results.mean) + '\n')
op2.write('b_std     ' + str(b_std))
op2.close()

print('Numerical results:')
print('a:    ' + str(a_results.mean) + ' +- ' + str(a_std))
print('b:    ' + str(b_results.mean) + ' +- ' + str(b_std))
print()


try:
    #plot results
    plot_2p(sampler_ABC.T, 'results.pdf' , Parameters)
except:
    print('Plotting ABC to results.pdf failed, maybe display not available.')

