import numpy as np
import pylab as plt

n_D = 750
sig2 = 5
r = [0, 0.5, 1, 2, 3, 4, 4.9]

cond_arr = []
for xcorr in r:
    cov = np.diag([sig2 - xcorr for i in range(n_D)]) + xcorr
    if xcorr == 0:
        Psi = np.diag([1.0 / sig2 for i in range(n_D)])
    else:
        c = 1.0/xcorr + n_D/(sig2 - xcorr)
        Psi = np.full((n_D, n_D), -1.0/c/(sig2 - xcorr)**2) \
                            + np.diag([1.0/(sig2 - xcorr) for i in range(n_D)])
    cond = np.linalg.cond(cov)
    cond_inv = np.linalg.cond(Psi)
    cond_arr.append(cond)

    print(f'{xcorr} {cond} {cond_inv}')
    #print(cov)
    #print(Psi)

plt.plot(r, cond_arr, 'x')
#plt.plot(r, [1/c for c in cond_arr], 'o')
plt.savefig('cond.pdf')
