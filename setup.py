#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

__name__ = 'CorrMatrix_ABC'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           __name__, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

setup(
    name = __name__,
    authors = release_info['__author__'],
    version = release_info['__version__'],
    url = 'https://github.com/emilleishida/CorrMatrix_ABC',
    packages = find_packages(),
    description = release_info['__description__'],
    install_requires = ['statsmodels', 'astropy', 'pystan'],
    scripts = ['scripts/cov_matrix_definition/cm_likelihood.py']
)
