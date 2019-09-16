#!/usr/bin/env python3

import numpy as np
import sys
import statsmodels.api as sm


from astropy.io import ascii
import pylab as plt

path = 'nsim_15/nr_4'

file_root = 'linear_PS'
iteration = 20

data_name = 'observation_xy.txt'

n = 2

# ABC sample file
file_path = '{}/{}{}.dat'.format(path, file_root, iteration)
samples = ascii.read(file_path)

np.random.seed(123)
rd = np.random.random(n) * len(samples)
idx = [int(f) for f in rd]

# Observed (simulated) data
data_path = '{}/{}'.format(path, data_name)
data = ascii.read(data_path)
x = data['col1']
y_data = data['col2']

fig, ax = plt.subplots(1, 1)

ax.plot(x, y_data-x, 'o', markersize=0.5)

data_obs = {}
data_obs['x'] = x
data_obs['y'] = y_data
data1_obs = np.array([[data_obs['x'][k], 1] for k in range(data_obs['x'].shape[0])])
mod_obs0 = sm.OLS(data_obs['y'], data1_obs)
mod_obs = mod_obs0.fit()
a_obs_ast = mod_obs.params[0]
b_obs_ast = mod_obs.params[1]
y_obs_ast = a_obs_ast * x + b_obs_ast 
ax.plot(x, y_obs_ast-x, '-', color='r', linewidth=1)
print(f'Best-fit a, b, obs = {a_obs_ast}, {b_obs_ast}')

a_mean = np.mean(samples['a'])
b_mean = np.mean(samples['b'])
a_std = np.std(samples['a'])
b_std = np.std(samples['b'])
ndraws_mean = np.mean(samples['NDraws'])

print(f'samples mean a, b, ndraws = {a_mean} {b_mean} {ndraws_mean}')
print(f'samples std a, b = {a_std} {b_std}')

for i in idx:
    a, b = samples['a'][i], samples['b'][i]
    y_sample = a * x + b 
    ax.plot(x, y_sample-x, '-', color='g', linewidth=0.5)
    print(samples[i])

    delta_a = a - a_obs_ast
    #delta_b = np.abs(b) - np.abs(b_obs_ast)
    delta_b = 0

    dist = np.sqrt( pow(delta_a, 2) + pow(delta_b, 2) )
    print(dist)

ax.set_xlabel('x')
ax.set_ylabel('y-x')

if len(sys.argv)>1:
    title_supp = ', {}'.format(sys.argv[1])
else:
    title_supp = ''

ax.set_title('iteration {}{}'.format(iteration, title_supp))

plt.savefig('samples.pdf')

fig, ax = plt.subplots(1, 1)
ax.hist(samples['a'], bins=20)
plt.savefig('hist_a.pdf')


