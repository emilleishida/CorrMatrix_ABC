import matplotlib

# The following line is required for the non-framework version of OSX python
matplotlib.use('TkAgg')
#matplotlib.use("Agg")

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from toy_model_functions import linear_dist_data_diag, linear_dist_data, linear_dist_data_acf, \
    linear_dist_data_acf_abs, linear_dist_data_acf_ratio, linear_dist_data_acf_ratio_abs, \
    model_cov, model_quad

import numpy as np
import sys

from scipy.stats import uniform, multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW
from covest import weighted_std, get_cov_WL

from astropy import units



#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['ampl'] = float(Parameters['ampl'][0])
Parameters['tilt'] = float(Parameters['tilt'][0])


# If prior is flat, does not need to be set (?; see abc_wl.py)

Parameters['f_sky']     = float(Parameters['f_sky'][0])
Parameters['sigma_eps'] = float(Parameters['sigma_eps'][0])
Parameters['nbar']      = float(Parameters['nbar'][0])

nell      = int(Parameters['nell'][0])
logellmin = float(Parameters['logellmin'][0])
logellmax = float(Parameters['logellmax'][0])
ellmode = Parameters['ellmode'][0]
if ellmode == 'log':
    # Equidistant in log ell
    logell = np.linspace(logellmin, logellmax, nell)
elif ellmode == 'lin':
    # Equidistant in ell
    ellmin = 10**logellmin
    ellmax = 10**logellmax
    ell = np.linspace(ellmin, ellmax, nell)
    logell = np.log10(ell)
else:
    raise ValueError('Invalid ellmode {}'.format(ellmode))

Parameters['simulation_func'] = model_cov

### Distance function
distance = {'linear_dist_data_diag':      linear_dist_data_diag,
            'linear_dist_data':           linear_dist_data,
            'linear_dist_data_acf':       linear_dist_data_acf,
            'linear_dist_data_acf_abs':   linear_dist_data_acf_abs,
            'linear_dist_data_acf_ratio': linear_dist_data_acf_ratio,
            'linear_dist_data_acf_ratio_abs': linear_dist_data_acf_ratio_abs,
           }
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

# true, fiducial model
u       = logell
y_true  = model_quad(u, Parameters['ampl'], Parameters['tilt'])

# Covariance
Parameters['nsim'] = int(Parameters['nsim'][0])
cov_model          = Parameters['cov_model'][0]
cov, cov_est       = get_cov_WL(cov_model, 10**logell, y_true, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'])

L = np.linalg.cholesky(cov)

input_is_true = int(Parameters['input_is_true'][0])
if input_is_true:
    y_input = y_true
else:
    # input model = sample from distribution with true covariance

    # Consistency check of input parameters
    path_to_obs = Parameters['path_to_obs']
    if path_to_obs != 'None':
        print('Inconsistent parameters: input_is_true = False (sampled input) *and* path_to_obs not \'None\'')
        sys.exit(5)

    y_input  = multivariate_normal.rvs(mean=y_true, cov=cov)

np.savetxt('y_true.txt', np.array([10**logell, y_true]).transpose())
np.savetxt('y_input.txt', np.array([10**logell, y_input]).transpose())

# add to parameter dictionary
Parameters['dataset1'] = np.array([[logell[i], y_input[i]] for i in range(nell)])
np.savetxt('dataset1.txt', Parameters['dataset1'], header='# ell C_ell')

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

#######################

# add covariance to user input parameters, to be used in model
#Parameters['cov'] = cov_est
Parameters['simulation_input']['cov'] = cov_est
np.savetxt('cov_est.txt', cov_est)

if distance_str == 'linear_dist_data':
    cov_true_inv = np.linalg.inv(cov)
    Parameters['cov_true_inv'] = cov_true_inv
    np.savetxt('cov_true_inv.txt', cov_true_inv)


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
op2.write('tilt_mean    ' + str(a_results.mean) + '\n')
op2.write('tilt_std     ' + str(a_results.std_mean) + '\n\n\n')
op2.write('ampl_mean    ' + str(b_results.mean) + '\n')
op2.write('ampl_std     ' + str(b_results.std_mean) + '\n')
op2.close()

print 'Numerical results:'
print 'tilt:    ' + str(a_results.mean) + ' +- ' + str(a_results.std_mean)
print 'ampl:    ' + str(b_results.mean) + ' +- ' + str(b_results.std_mean)

# Write best-fit model to file
y_bestfit   = model_quad(u, b_results.mean, a_results.mean)
np.savetxt('y_bestfit.txt', np.array([10**logell, y_bestfit]).transpose())

#plot results
out_path = 'results.pdf'
try:
    plot_2p(sampler_ABC.T, out_path, Parameters)
except:
    print('Error occured while creating \'{}\'. Maybe just the display could not be accessed. Continuing anyway...'.format(out_path))
