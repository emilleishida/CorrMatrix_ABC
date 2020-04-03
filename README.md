# CorrMatrix_ABC
Study of correlation matrices under ABC.

## Use and run the codes

The following instruction lets you run the codes and scripts that come with this package.
You can reproduce the plots and results from the paper.

## Install ```cosmoabc```

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
Optionally get `jupyter` to open and run notebooks:
```bash
conda install -c conda-forge jupyterlab
```

Install `cosmoabc` with `pip`.
```
pip install cosmoabc
```
If this fails due to a python error, try to install `cosmoabc` from `github`:
```bash
git clone git@github.com:COINtoolbox/CosmoABC.git
cd CosmoABC
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
export PYTHONPATH=CorrMatrix_ABC/scripts
```

## Installation for MCMC sampling

You need pystan to run the MCMC part, with a specific pystan version:
```bash
conda install pystan=2.18
```

The main program for MCMC sampling of the example models from the paper is `cm_likelihood.py`.
Run `cm_likelihood.py --help` to see the options.

## Toy example 1

### The model set-up

This example is an affine function y=ax+b with fiducial parameter a=1, b=0. We use n_d = 750 (option `-D`) data points, drawn from a multi-variate normal with diagonal covariance with constant sigma=5 (`-s`) on the diagonal.

The number of simulations n_s used to compute the covariance can be specified by the user. A number of different cases can be run sequential, and results are plotted as function of n_s. These numbers can be given as array with `--n_S`, e.g. `--n_S 755_1500_3000`, which creates simulations for three different n_s, 755, 1500, and 3000.
Without this option, the default is 10 logarithmically spaced numbers between n_d + 5 and 10 n_d. When in doubt, run with `-m d` to display the number of simulations to be used without doing the sampling.

### Fisher matrix

To start, let us only compute the Fisher-matrix predictions without MCMC, which is significantly faster.
First, we create the simulations (`-m s`), each of the 10 cases of n_s with n_r=50 runs.

```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 50
```

This outputs results for the multi-variate normal likelihood with biased (infix `_num`) and debiased precision matrix (`_deb`). The plots show the parameter RMS (prefix `std_Fisher`) and RMS of the parameter variances (`std_2std_Fisher`).
In addition, plots of the trace of the covariance maximum-likelihood estimate
(`sigma_ML`) and precision matrix, biased (`sigma_m1_ML`) and debiased (`sigma_m1_ML_deb`) are produced.

### MCMC sampling

#### Normal likelihood

To run the sampling with `pystan`, we need to add the option `--fit_stan`, and specify the likelihood. Let's use the Gaussian  with debiased covariance (`-L norm_deb`). To save time, we reduce the number of runs, say to n_r=5.

First, we create the simulations (`-m s`), with results automatically written in text files on disk.
```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 5--fit_stan -L norm_deb --plot_style paper
```
More simulation runs can always be added to existing ones with the option `-a`. Let's create and add additional 3 runs: 
```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -a -R 3 --fit_stan -L norm_deb --plot_style paper
```
At the end, to read all existing runs and create plots, use the *read* mode with `-m r`:
```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m r -r -R 8 --fit_stan -L norm_deb --plot_style paper
```
Since this does not run simulations, this step is fast.

The 50 runs used for Figs. 2 and 4 from the paper are available with this package, to reproduce the plots:
```bash
cd results/norm_deb_MCMC
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m r -r -R 50 --fit_stan -L norm_deb --sig_var_noise 4.6e-08_0.000175 --plot_style paper
```
There are additional options: The raw Fisher-matrix-predicted parameter variance RMS under-estimates the MCMC result. The paper claims that this is due to inherent MCMC "noise". This noise was estimated by running MCMC with the true precision matrix, to isolate this effect. We can subtract this noise from the data points by adding the option `--sig_var_noise 4.6e-08_0.000175`, and indeed the corrected prediction matches much better. The `--plot_style` options can be `paper` or `talk`, producing small changes in the layout.

#### Hotelling T^2 likelihood

Hotelling (1931), also derived in Sellentin & Heavens (2016).
Instead of `-L norm_deb`, use the flag `-L SH` for this modified normal likelihood that accounts for the uncertainty in the sample covariance. Otherwise run the script `cm_likelihood.py` as for the normal case above.

To reproduce Figs. 3 and 5 from the paper:
```
cd results/T2_MCMC
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m r -R 50 --fit_stan -L SH --plot_style paper
```

### ABC

The main job user-control script for ABC is in the directory `scripts/ABC'. To create a single ABC run, type
```bash
job_ABC.py --n_S 125 -v -m r --template_dir templates -b 0.15 --xlog -R 1
```
As before for MCMC, the options `--n_S` and `-R` indicate the number of simulations for the covariance, and the number of independent runs, respectively.

The required parameters and python functions to run ABC are found in the template directory, indicated by the `--template_dir` option. This can be a link to the default files `scripts/ABC/template`. Three files are read:
  1. `toy_model.input`
    The ABC configuration file. See https://github.com/COINtoolbox/CosmoABC for more details. Here, we just emphazise a few config entries:  
    - `nsim = 800`: This is the number of simulations for the covariance, the value will be overwritten by the `--n_S` option.
  2. `toy_model_functions.py`: This python script contains some functions that are specified in the `toy_model.input` configuration file. In particular, these are the functions to create the simulation (config entry `simulation_func`), the distance (`distance_func`), and parameter prior (`prior_func`).
  3. `ABC_est_cov.py`: The master executable python program. It reads the config file, initalises all parameters, creates the covariance matrix, runs ABC, and outputs statistics and plots. This script is called by `job_ABC.py`.
  
The command above creates ABC output files in the directory `nsim_125/nr_0`. More runs (with `-R <n_r>' with n_r>1)
will be written to `nsim_125/nr_1`, `nsim_125/nr_2`, etc. As before, multiple n_s values can be given, each case will be written in the respective `nsim_<n_s>` directory. See https://github.com/COINtoolbox/cosmoabc for a description of the output files.

Fig. 6 can in principle be reproduced with the command `job_ABC.py --n_S 2_5_26_15_58_125_755_1622_3488_7500 -v -m s --template_dir templates -b 0.15 --xlog -R 50`. Since this will however take a long time, we have added to this repository all ABC output result files `num_res.dat`. They can be read in with
```bash
cd results/norm_ABC
job_ABC.py --n_S 2_5_26_15_58_125_755_1622_3488_7500 -v -m s --template_dir templates -b 0.15 --xlog -R 50
```
