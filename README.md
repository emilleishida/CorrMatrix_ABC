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
'''

Install `cosmoabc` with `pip`.
```
pip install cosmoabc
```

#### MCMC sampling

You need pystan to run the MCMC part, with a specific pystan version:

```bash
conda install pystan=2.18
```
