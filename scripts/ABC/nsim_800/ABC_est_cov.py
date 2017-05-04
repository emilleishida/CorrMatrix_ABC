from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from toy_model_functions import model, linear_dist

import numpy as np
from scipy.stats import uniform, multivariate_normal

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


#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['xmin'] = float(Parameters['xmin'][0])
Parameters['xmax'] = float(Parameters['xmax'][0])
Parameters['a'] = float(Parameters['a'][0])
Parameters['b'] = float(Parameters['b'][0])
Parameters['sig'] = float(Parameters['sig'][0])
Parameters['nobs'] = int(Parameters['nobs'][0])
Parameters['nsim'] = int(Parameters['nsim'][0])


# set functions
Parameters['simulation_func'] = model
Parameters['distance_func'] = linear_dist

# construnct 1 instance of exploratory variable
x = uniform.rvs(loc=Parameters['xmin'], scale=Parameters['xmax'] - Parameters['xmin'], size=Parameters['nobs'])
x.sort()

# fiducial model
ytrue = Parameters['a']*x + Parameters['b']

# real covariance matrix
cov = np.diag([Parameters['sig'] for i in range(Parameters['nobs'])]) 

# generate catalog
y = multivariate_normal.rvs(mean=ytrue, cov=cov, size=1)

# add to parameter dictionary
Parameters['dataset1'] = np.array([[x[i], y[i]] for i in range(Parameters['nobs'])])


#estimated covariance matrix
cov_est = get_cov_ML(ytrue, cov, Parameters['nsim'])

# add covariance to user input parameters
Parameters['cov'] = cov_est


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

a_mean = sum([a_samples[i] * weights[i] for i in range(sampler_ABC.M)])

nonzerow = sum([1 for item in weights if item > 0])
a_std = sum([weights[i]*(a_samples[i] - a_mean)/(((nonzerow - 1)/float(nonzerow))*sum(weights))])

b_mean = sum([b_samples[i] * weights[i] for i in range(sampler_ABC.M)])
b_std = sum([weights[i]*(b_samples[i] - b_mean)/(((nonzerow - 1)/float(nonzerow))*sum(weights))])

# store numerical results
op2 = open('num_res_nsim_' + str(Parameters['nsim']) + '.dat', 'w')
op2.write('a_mean    ' + str(a_mean) + '\n')
op2.write('a_std     ' + str(a_std) + '\n\n\n')
op2.write('b_mean    ' + str(b_mean) + '\n')
op2.write('b_std     ' + str(b_std))
op2.close()

print 'Numerical results:'
print 'a:    ' + str(a_mean) + ' +- ' + str(a_std)
print 'b:    ' + str(b_mean) + ' +- ' + str(b_std)



#plot results
plot_2p( sampler_ABC.T, 'results_nsim_' + str(Parameters['nsim']) + '.pdf' , Parameters)





