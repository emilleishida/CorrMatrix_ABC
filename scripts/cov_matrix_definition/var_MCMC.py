#!/usr/bin/env python

import numpy as np
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.base.model import LikelihoodModel
import pylab as plt
import matplotlib.mlab as mlab

from covest import *

def get_var_flattened(res):

    n_n_S, n_R = res.mean[res.par_name[0]].shape

    all_var = {}
    for p in res.par_name:
        all_var[p] = []

        for i in range(n_n_S):
            for run in range(n_R):
                all_var[p].append(res.std[p][i][run]**2)

    return all_var



def all_var_print(var, par_name, file_base):

    for p in par_name:
        mean_var = np.mean(var[p])
        std_var  = np.std(var[p])
        print('{:30s} {:3s} {:.5g} {:.5g}'.format(file_base, p, mean_var, std_var))



class Norm(GenericLikelihoodModel):

    nparams = 3

    def loglike(self, params):
        return norm.logpdf(self.endog, *params).sum()


def fit_plot_norm(var, hist, col):

    params = np.array([0.0, 1.0])

    res = Norm(var).fit(start_params=params, disp=0)
    mu        = res.params[0]
    sigma     = res.params[1]
    if res.normalized_cov_params is not None:
        std_mu    = np.sqrt(res.normalized_cov_params[0][0])
        std_sigma = np.sqrt(res.normalized_cov_params[1][1])
    else:
        std_mu    = 0
        std_sigma = 0

    #print(res.__dict__)
    #print(dir(res))

    # This gives an error
    #res.df_model = len(params)
    #res.df_resid = len(var) - len(params)
    #print(res.summary())

    bins = hist[1]
    dx = (bins[1]-bins[0])
    x = bins + dx/2
    x = x[:-1]

    nf = 2
    n = np.linspace(x[0] - nf*dx, x[-1] + nf*dx, num=len(x)*10)

    y = mlab.normpdf(n, mu, sigma)
    plt.plot(n, y, '{}-'.format(col), linewidth=2)

    return mu, sigma, std_mu, std_sigma



def main(argv=None):
    """Main program.
    """

    par_name = ['a', 'b']

    n_n_S    =  4
    n_R      = 12
    fit_true_cov = Results(par_name, n_n_S, n_R, file_base='true_cov/mean_std_fit_norm')
    fit_true_cov.read_mean_std(verbose=True)
    var_true_cov = get_var_flattened(fit_true_cov)
    print('Number of runs for {} = {}'.format(fit_true_cov.file_base, len(var_true_cov[par_name[0]])))

    n_n_S    =  6
    n_R      = 19
    fit_true_inv_cov = Results(par_name, n_n_S, n_R, file_base='true_inv_cov/mean_std_fit_norm')
    fit_true_inv_cov.read_mean_std(verbose=True)
    var_true_inv_cov = get_var_flattened(fit_true_inv_cov)
    print('Number of runs for {} = {}'.format(fit_true_inv_cov.file_base, len(var_true_inv_cov[par_name[0]])))

    print('# {:28s} {:3s} {:12s} {:12s}'.format('file', 'par', 'mean(var)', 'std(var)'))
    all_var_print(var_true_cov, par_name, fit_true_cov.file_base)
    all_var_print(var_true_inv_cov, par_name, fit_true_inv_cov.file_base)

    for p in par_name:

        fig = plt.figure()
        ax = plt.gca().xaxis
        ax.set_major_formatter(ScalarFormatter())

        hist_true_cov = plt.hist(var_true_cov[p], alpha = 0.5, normed=True)
        hist_true_inv_cov = plt.hist(var_true_inv_cov[p], alpha = 0.5, normed=True)

        mu, sigma, mu_std, sigma_std = fit_plot_norm(var_true_cov[p], hist_true_cov, 'g')
        print('fit {:35s} par={:3s} sigma={:.3e} +- {:.3e}'.format(fit_true_cov.file_base, p, sigma, sigma_std))

        mu, sigma, mu_std, sigma_std = fit_plot_norm(var_true_inv_cov[p], hist_true_inv_cov, 'b')
        print('fit {:35s} par={:3s} sigma={:.3e} +- {:.3e}'.format(fit_true_inv_cov.file_base, p, sigma, sigma_std))

        if p == 'a':
            plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.xlabel('$\\sigma^2({})$'.format(p))
        plt.ylabel('normalised frequency')
        plt.title('Parameter {}'.format(p))

        plt.savefig('hist_std_var_{}.pdf'.format(p))




if __name__ == "__main__":
    sys.exit(main(sys.argv))




