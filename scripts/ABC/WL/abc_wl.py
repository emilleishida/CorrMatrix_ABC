#!/usr/bin/env python

"""abc_wl.py

Template for python script.
TODO: Copy content ABC_cov_est.py here.

:Author: Martin Kilbinger

*Date*: 2015

"""


"""Thoughts.

Create simulation as model:
    - For given parameters, get theoretical power spectrum C_l. Then,
        - either simulate shear/kappa Gaussian field with C_l
          as (co-)variance, measure hat C_l or hat xi_+- = model.
        - or simulate hat C_l as Wishart using C_l as input.

Input:
    - Survey: size, number of galaxies, (mask),
      redshift distribution (sample from it for model!), number of z-bins,
      angular scales.
    - Parameters, prior values
"""

# Compability with python2.x for x>6
from __future__ import print_function


import sys
import os
import copy

import numpy as np
import pylab as plt

from astropy.io import ascii
from astropy.table import Table, Column

from optparse import OptionParser
from optparse import OptionGroup

import mkstuff
import mkplot



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class mkstuff.param
        parameter values
    """

    p_def = mkstuff.param(
    )

    return p_def



def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class mkstuff.param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')

    options, args = parser.parse_args()

    return options, args



def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    see_help = 'See option \'-h\' for help.'

    return True



def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class mkstuff.param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class mkstuff.param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    # Do extra stuff if necessary
    # ...

    return param



def main(argv=None):
    """Main program.
    """

    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)


    # Save calling command
    mkstuff.log_command(argv)
    if param.verbose:
        mkstuff.log_command(argv, name='sys.stderr')


    if param.verbose is True:
        print('Start program {}'.format(os.path.basename(argv[0])))


    ### Start main program ###




    ### End main program

    if param.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

