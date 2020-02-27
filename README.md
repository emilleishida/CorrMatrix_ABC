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

The main program for MCMC sampling of the example models from the paper is `scripts/cov_matrix_definition/test_ML.py`.
Run `test_ML.py --help` to see the options.

### Toy model 1

test_ML.py -D 750 -p 1_0 -s 5 -v -m r -r -R 50 --fit_stan -L norm_deb --sig_var_noise 4.6e-08_0.000175 --plot_style paper


