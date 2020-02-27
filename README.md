# CorrMatrix_ABC
Study of correlation matrices under ABC.

## Reproduce results from the paper.

### Install ```cosmoabc```

Use a `conda` environment:
```bash
conda create -n ABC python=2.7
conda activate ABC
conda install statsmodels ipython
```

The following packages should be installed by default, if not, install them by hand:
```bash
conda install numpy scipy matplotlib astropy
```

Install `cosmoabc` with `pip`.
```
pip install cosmoabc
```
If this fails due to a python error, try to install `cosmoabc from `github`:
```bash
git clone git@github.com:emilleishida/CorrMatrix_ABC.git
```
Add square brackets to the line in `setup.py` as follows:
`package_data = {'cosmoabc/data':['SPT_sample.dat']},`
and install the package
```bash
python2 setup.py install
```

At the last installation step, clone this package:
```bash
git clone git@github.com:emilleishida/CorrMatrix_ABC.git
```

#### MCMC sampling

You need pystan to run the MCMC part, with a specific pystan version:
```bash
conda install pystan=2.18
```
