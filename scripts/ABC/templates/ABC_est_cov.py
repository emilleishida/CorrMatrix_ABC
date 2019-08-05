import sys

import matplotlib
matplotlib.use("TkAgg")

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from toy_model_functions import model, linear_dist, linear_dist_data, model, model_cov, gaussian_prior

import numpy as np
from scipy.stats import uniform, multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW
from covest import weighted_std


#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['xmin'] = float(Parameters['xmin'][0])
Parameters['xmax'] = float(Parameters['xmax'][0])
Parameters['a'] = float(Parameters['a'][0])
Parameters['b'] = float(Parameters['b'][0])
Parameters['sig'] = float(Parameters['sig'][0])
Parameters['xcorr'] = float(Parameters['xcorr'][0])
Parameters['nobs'] = int(Parameters['nobs'][0])

# set functions
Parameters['simulation_func'] = model_cov
Parameters['distance_func'] = linear_dist
#Parameters['distance_func'] = linear_dist_data
Parameters['prior']['a']['func'] = gaussian_prior
Parameters['prior']['b']['func'] = gaussian_prior

# construnct 1 instance of exploratory variable
x = uniform.rvs(loc=Parameters['xmin'], scale=Parameters['xmax'] - Parameters['xmin'], size=Parameters['nobs'])
x.sort()

# fiducial model
ytrue = Parameters['a']*x + Parameters['b']

# true covariance matrix: *sig* on diagonal, *xcorr* on off-diagonal
sig2 = Parameters['sig']
xcorr = Parameters['xcorr']
cov = np.diag([sig - xcorr for i in range(Parameters['nobs'])]) + xcorr

# check whether cov is positive definite
if xcorr != 0:
    L = np.linalg.cholesky(cov)
sys.exit(0)

#############################################
Parameters['nsim'] = int(Parameters['nsim'][0])
cov_est = get_cov_ML(ytrue, cov, Parameters['nsim'])

# add covariance to user input parameters
Parameters['simulation_input']['cov'] = cov_est

# save cov_est.txt to disk
# this file is read when running continue_ABC.py or plot_ABC.py
np.savetxt('cov_est.txt', cov_est)

if len(sys.argv) > 1 and sys.argv[1] == '--only_cov_est':
    print('Written estimated covariance matrix, exiting.')
    sys.exit(0)
#############################################


# generate catalog
y = multivariate_normal.rvs(mean=ytrue, cov=cov, size=1)

# add to parameter dictionary
Parameters['dataset1'] = np.array([[x[i], y[i]] for i in range(Parameters['nobs'])])
# write to disk, in case it is read by continue_ABC.py
op1 = open('observation_xy.txt', 'w')
for line in Parameters['dataset1']:
    for item in line:
        op1.write(str(item) + '    ')
    op1.write('\n')
op1.close()

# add observed catalog to simulation parameters
if bool(Parameters['xfix']):
    Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

#initiate ABC sampler
sampler_ABC = ABC(params=Parameters)

#build first particle system
sys1 = sampler_ABC.BuildFirstPSystem()

#update particle system until convergence
nruns = int(Parameters['nruns'][0])
sampler_ABC.fullABC(nruns=nruns)


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
