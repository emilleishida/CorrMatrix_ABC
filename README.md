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

### MCMC sampling

You need pystan to run the MCMC part, with a specific pystan version:
```bash
conda install pystan=2.18
```

The main program for MCMC sampling of the example models from the paper is `cm_likelihood.py`.
Run `cm_likelihood.py --help` to see the options.

### Toy example 1

This example is an affine function y=ax+b with fiducial parameter a=1, b=0. A pystan fit (`--fit_stan`) is performed using n_d = 750 (option `-D`) data points, drawn from a multi-variate normal with diagonal covariance with constant sigma=5 (`-s`) on the diagonal. The model is `norm_deb` (`-L`), indicating a Gaussian likelihood with debiased covariance matrix. The paper shows the distribution of n_r=50 (`-R`) runs. Since this takes a long time to run, a smaller number can be used, say 5.

The number of simulations n_s used to compute the covariance can be specified by the user. A number of different cases can be run sequential, and results are plotted as function of n_s. These numbers can be given as array with `--n_S`, e.g. `--n_S 755_1500_3000`, which creates simulations for three different n_s, 755, 1500, and 3000.
Without this option, the default is 10 logarithmically spaced numbers between n_d + 5 and 10 n_d.

First, we create the simulations (`-m s`).

```bash
cm_likelihood.py -D 750 -p 1_0 -s 5 -v -m s -r -R 5 --n_n_S 10 --fit_stan -L norm_deb --sig_var_noise 4.6e-08_0.000175 --plot_style paper
```

