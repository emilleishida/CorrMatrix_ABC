import matplotlib
#matplotlib.use("Agg")

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from toy_model_functions import linear_dist_data_diag, linear_dist_data, model_cov, quadratic_amp_tilt

import numpy as np
from scipy.stats import uniform, multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW
from covest import weighted_std, get_cov_WL

from astropy import units



#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['amp'] = float(Parameters['amp'][0])
Parameters['tilt'] = float(Parameters['tilt'][0])


# If prior is flat, does not need to be set (?; see abc_wl.py)

Parameters['f_sky']     = float(Parameters['f_sky'][0])
Parameters['sigma_eps'] = float(Parameters['sigma_eps'][0])
Parameters['nbar']      = float(Parameters['nbar'][0])

# construnct 1 instance of exploratory variable
logellmin = float(Parameters['logellmin'][0])
logellmax = float(Parameters['logellmax'][0])
nell      = int(Parameters['nell'][0])
logell    = np.linspace(logellmin, logellmax, nell)


Parameters['simulation_func'] = model_cov

### Distance function
distance = {'linear_dist_data_diag': linear_dist_data_diag,
            'linear_dist_data':      linear_dist_data,
           }
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

# fiducial model
Cell_true  = quadratic_amp_tilt(logell, Parameters['amp'], Parameters['tilt'])

# add to parameter dictionary
Parameters['dataset1'] = np.array([[logell[i], Cell_true[i]] for i in range(nell)])

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

#######################
# Covariance

Parameters['nsim'] = int(Parameters['nsim'][0])
cov, cov_est = get_cov_WL('Gauss', 10**logell, Cell_true, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'])


# add covariance to user input parameters, to be used in model
Parameters['cov'] = cov_est
Parameters['simulation_input']['cov'] = cov_est

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
op2.write('a_mean    ' + str(a_results.mean) + '\n')
op2.write('a_std     ' + str(a_results.std_mean) + '\n\n\n')
op2.write('b_mean    ' + str(b_results.mean) + '\n')
op2.write('b_std     ' + str(b_results.std_mean) + '\n')
op2.close()

print 'Numerical results:'
print 'a:    ' + str(a_results.mean) + ' +- ' + str(a_results.std_mean)
print 'b:    ' + str(b_results.mean) + ' +- ' + str(b_results.std_mean)



#plot results
plot_2p( sampler_ABC.T, 'results.pdf' , Parameters)
