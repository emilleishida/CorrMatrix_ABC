from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
import numpy as np
from cosmoabc.ABC_functions import read_input

from toy_model_functions import sim_linear_ind, linear_dist, model

#user input file
filename = 'toy_model.input'

#read  user input
Parameters = read_input(filename)

#initiate ABC sampler
sampler_ABC = ABC(params=Parameters)

#build first particle system
sys1 = sampler_ABC.BuildFirstPSystem()

#update particle system until convergence
sampler_ABC.fullABC()

#plot results
plot_2D( sampler_ABC.T, 'results.pdf' , params)
