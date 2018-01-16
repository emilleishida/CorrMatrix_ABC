#!/usr/bin/env python

# trim_runs.py
# Martin Kilbinger 2017


# Compability with python2.x for x>6
from __future__ import print_function


import sys
import os
import copy
import re

import numpy as np

from astropy.io import ascii
from astropy.table import Table, Column

from optparse import OptionParser
from optparse import OptionGroup

from covest import *



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class param
        parameter values
    """

    p_def = param(
        n_R = 10,
        in_base = 'mean_std_fit_norm',
    )

    return p_def



def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class param
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

    parser.add_option('-i', '--in_name', dest='in_base', type='string', default=p_def.in_base,
        help='Input file name base (without extension), default={}'.format(p_def.in_base))
    parser.add_option('-o', '--out_name', dest='out_base', type='string',
        help='Output file name base (without extension), default=<in_base>_R_<R>')

    parser.add_option('-R', '--n_R', dest='n_R', type='int', default=p_def.n_R,
        help='Number of runs per simulation to trim, default={}'.format(p_def.n_R))

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
    p_def:  class param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class param
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
    if param.out_base == None:
        param.out_base = '{}_R_{}'.format(param.in_base, param.n_R)

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
    log_command(argv)
    if param.verbose:
        log_command(argv, name='sys.stderr')


    if param.verbose is True:
        print('Start program {}'.format(os.path.basename(argv[0])))


    ### Start main program ###

    par_name = ['a', 'b']            # Parameter list


    n_S_arr, n_R_prev = get_n_S_R_from_fit_file(param.in_base, npar=len(par_name))
    n_n_S = len(n_S_arr)
    n_R_new = param.n_R

    if n_R_new > n_R_prev:
        error('New number of runs {} is larger than previous one {}'.format(n_R_new, n_R_prev))

    fit = Results(par_name, n_n_S, n_R_new, file_base=param.in_base)
    fit.read_mean_std(verbose=param.verbose)

    fit.file_base = param.out_base

    fit.write_mean_std(n_S_arr)


    ### End main program

    if param.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

