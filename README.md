# CorrMatrix_ABC
Study of correlation matrices under ABC.

## Use and run the codes

The following instruction lets you run the codes and scripts that come with this package.
You can reproduce the plots and results from the paper.

## Install ```cosmoabc```

Use a `conda` environment:
```bash
conda env create -f environment.yml
conda activate corrABC
python setup.py install
```

The main program for Fisher-matrix computation and MCMC sampling of the first
two example models
from the paper is `cm_likelihood.py`. Run `cm_likelihood.py --help` to see the options.

## Example 1

Affine function with diagonal covariance matrix.

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

First, we need to install the additional `pystan` package:
```
conda install pystan=2.18
```

To run the sampling we add the option `--fit_stan`, and specify the likelihood. Let's use the Gaussian  with debiased covariance (`-L norm_deb`). To save time, we reduce the number of runs, say to n_r=5.

First, we create the simulations (`-m s`), with results automatically written in text files on disk.
```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 5 --fit_stan -L norm_deb --plot_style paper
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

The 50 runs used for the paper are available with this package, to reproduce the plots:
```bash
cd results/norm_deb_MCMC
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m r -r -R 50 --fit_stan -L norm_deb --sig_var_noise 4.6e-08_0.000175 --plot_style paper
```
There are additional options: The raw Fisher-matrix-predicted parameter variance RMS under-estimates the MCMC result. The paper claims that this is due to inherent MCMC "noise". This noise was estimated by running MCMC with the true precision matrix, to isolate this effect. We can subtract this noise from the data points by adding the option `--sig_var_noise 4.6e-08_0.000175`, and indeed the corrected prediction matches much better. The `--plot_style` options can be `paper` or `talk`, producing small changes in the layout.

#### Hotelling T^2 likelihood

Hotelling (1931), also derived in Sellentin & Heavens (2016).
Instead of `-L norm_deb`, use the flag `-L SH` for this modified normal likelihood that accounts for the uncertainty in the sample covariance. Otherwise run the script `cm_likelihood.py` as for the normal case above.

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
cd results/Example_1/norm_ABC
job_ABC.py --n_S 2_5_26_15_58_125_755_1622_3488_7500 -v -m r --template_dir ../../../scripts/ABC/templates -b 0.15 --xlog -R 50
```

### Combined plots

Figure 3 in the paper shows results for ABC, and MCMC sampling under the normal and Hotelling T^2 likelihood.
To reproduce the figure without recomputing all sampling results, do the following steps:
```bash
cd results/Example_1
../../scripts/plot_many.py --ABC norm_ABC/mean_std_ABC --MCMC_norm norm_deb_MCMC/mean_std_fit_norm_deb --MCMC_T2 T2_MCMC/mean_std_fit_SH -b 0.03 --sig_var 4.6e-8_1.75e-4
```

## Example 2

Weak-lensing inspired case.

Figure 4 shows a weak-lensing power spectrum and the quadratic function fit. Three cases of three cosmological parametere are plotted.
This figure is reproduced with the jupyter notebook [WL_Cell_fit.ipynb](scripts/ABC/WL_Cell_fit.ipynb).

Figure 6 displays ABC results with the acf distance. To reproduce this plot, type:
```bash
cd results/Example_2/Euclid_dist_acf2lin
../../../scripts/ABC/job_ABC.py -M quadratic --template_dir ../../../scripts/ABC/templates_quad -m r --n_S 2_5_10_20_40 -R 25 --obs_dir nsim_2 -v -p 0.306_0.827 -P tilt_ampl -D 10
```

## Example 3

Realistic weak-lensing case.

Weak-lensing power spectrum models are produced with `nicaea`. This package has to be installed first, see https://github.com/martinkilbinger/nicaea .

Figure 7 shows ABC results with the acf distance. To reproduce this figure, type:
```bash
cd results/Example_3/Euclid_dist_acf2lin
../../../scripts/ABC/job_ABC.py -M wl --template_dir ../../../scripts/ABC/WL -m r --n_S 2_5_10_20_40 -R 10 --obs_dir nsim_2 -v -p 0.306_0.827 -P Omegam_sigma8 -D 10 

