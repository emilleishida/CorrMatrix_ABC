# CorrMatrix_ABC
Study of correlation matrices under ABC.

## Use and run the codes

The following instruction lets you run the codes and scripts that come with this package.
You can reproduce the plots and results from the paper.

### Install ```cosmoabc```

Use a `conda` environment:
```bash
conda create -n ABC python=2.7
conda activate ABC
conda install statsmodels ipython astropy
```

The following packages should be installed by default, if not, install them by hand:
```bash
conda install numpy scipy matplotlib
```

Install `cosmoabc` with `pip`.
```
pip install cosmoabc
```
If this fails due to a python error, try to install `cosmoabc` from `github`:
```bash
git clone git@github.com:emilleishida/CorrMatrix_ABC.git
```
Add square brackets to the line in `setup.py` as follows:
`package_data = {'cosmoabc/data':['SPT_sample.dat']},`
and install the package
```bash
python2 setup.py install
```

At the last installation steps, clone this package:
```bash
git clone git@github.com:emilleishida/CorrMatrix_ABC.git
```
and set the `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:CorrMatrix_ABC/scripts
```

### MCMC sampling and Fisher matrix

You need pystan to run the MCMC part, with a specific pystan version:
```bash
conda install pystan=2.18
```

The main program for MCMC sampling of the example models from the paper is `cm_likelihood.py`.
Run `cm_likelihood.py --help` to see the options.

### Toy example 1

#### The model set-up

This example is an affine function y=ax+b with fiducial parameter a=1, b=0. We use n_d = 750 (option `-D`) data points, drawn from a multi-variate normal with diagonal covariance with constant sigma=5 (`-s`) on the diagonal.

The number of simulations n_s used to compute the covariance can be specified by the user. A number of different cases can be run sequential, and results are plotted as function of n_s. These numbers can be given as array with `--n_S`, e.g. `--n_S 755_1500_3000`, which creates simulations for three different n_s, 755, 1500, and 3000.
Without this option, the default is 10 logarithmically spaced numbers between n_d + 5 and 10 n_d. When in doubt, run with `-m d` to display the number of simulations to be used without doing the sampling.

#### Fisher matrix

To start, let us only compute the Fisher-matrix predictions without MCMC, which is significantly faster.
First, we create the simulations (`-m s`), each of the 10 cases of n_s with n_r=50 runs.

```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 50 --n_n_S 10
```

This outputs results for the multi-variate normal likelihood with biased (infix `_num`) and debiased precision matrix (`_deb`). The plots show the parameter RMS (prefix `std_Fisher`) and RMS of the parameter variances (`std_2std_Fisher`).
In addition, plots of the trace of the covariance maximum-likelihood estimate
(`sigma_ML`) and precision matrix, biased (`sigma_m1_ML`) and debiased (`sigma_m1_ML_deb`) are produced.

#### MCMC sampling

To run the sampling with `pystan`, we need to add the option `--fit_stan`, and specify the likelihood. Let's use the Gaussian  with debiased covariance (` -L norm_deb`). To save time, we reduce the number of runs, say to n_r=5.

First, we create the simulations (`-m s`).

```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 5 --n_n_S 10 --fit_stan -L norm_deb --plot_style paper
```

Fig. 4 in the paper shows that the raw Fisher-matrix-predicted parameter variance RMS under-estimates the MCMC result. The paper claims that this is due to inherent MCMC "noise". This noise was estimated by running MCMC with the true precision matrix, to isolate this effect. We can subtract this noise from the data points by adding the option `--sig_var_noise 4.6e-08_0.000175`, and indeed the corrected prediction matches much better.


