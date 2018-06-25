import numpy as np
from statsmodels.tsa.stattools import acf

n = 50
lmin = 2

l     = np.arange(2, 2+n)
clth  = l**1.5
sigma = 5 

cl = np.random.normal(loc=clth, scale=sigma, size=n)

print(l)
print(cl)
