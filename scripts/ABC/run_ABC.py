#!/usr/bin/env python

# Compability with python2.x for x>6
from __future__ import print_function


import sys
import os
import re
import subprocess

import shlex
from shutil import copy2

import numpy as np

sys.path.append('..')
sys.path.append('../..')
from covest import *



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

    p_def = param(
        n_D = 750,
        n_R = 4,
        n_n_S = 10,
        f_n_S_max = 10,
        spar = '1.0 0.0',
        sig2 = 5.0,
        verbose = True,
        templ_dir = 'templates',
        mode = 'r',
    )

    return p_def




def run_cmd(cmd_list, run=True, verbose=True, stop=False, parallel=True, file_list=None, devnull=False):
    """Run shell command or a list of commands using subprocess.Popen().

    Parameters
    ----------

    cmd_list: string, or array of strings
        list of commands
    run: bool
        If True (default), run commands. run=False is for testing and debugging purpose
    verbose: bool
        If True (default), verbose output
    stop: bool
        If False (default), do not stop after command exits with error.
    parallel: bool
        If True (default), run commands in parallel, i.e. call subsequent comands via
        subprocess.Popen() without waiting for the previous job to finish.
    file_list: array of strings
        If file_list[i] exists, cmd_list[i] is not run. Default value is None
    devnull: boolean
        If True, all output is suppressed. Default is False.

    Returns
    -------
    sum_ex: int
        Sum of exit codes of all commands
    """

    if type(cmd_list) is not list:
        cmd_list = [cmd_list]

    if verbose is True and len(cmd_list) > 1:
        print('Running {} commands, parallel = {}'.format(len(cmd_list), parallel))


    ex_list   = []
    pipe_list = []
    for i, cmd in enumerate(cmd_list):

        ex = 0

        if run is True:

            # Check for existing file
            if file_list is not None and os.path.isfile(file_list[i]):
                if verbose is True:
                    print_color('blue', 'Skipping command \'{}\', file \'{}\' exists'.format(cmd, file_list[i]))
            else:
                if verbose is True:
                        print_color('green', 'Running command \'{0}\''.format(cmd))

                # Run command
                try:
                    cmds = shlex.split(cmd)
                    if devnull is True:
                        pipe = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
                    else:
                        pipe = subprocess.Popen(cmds)

                    if parallel is False:
                        # Wait for process to terminate
                        pipe.wait()

                    pipe_list.append(pipe)

                    # If process has not terminated, ex will be None
                    #ex = pipe.returncode
                except OSError as e:
                    print_color('red', 'Error: {0}'.format(e.strerror))
                    ex = e.errno

                    check_error_stop([ex], verbose=verbose, stop=stop)

        else:
            if verbose is True:
                print_color('yellow', 'Not running command \'{0}\''.format(cmd))

        ex_list.append(ex)


    if parallel is True:
        for i, pipe in enumerate(pipe_list):
            pipe.wait()

            # Update exit code list
            ex_list[i] = pipe.returncode


    s = check_error_stop(ex_list, verbose=verbose, stop=stop)

    return s


def check_error_stop(ex_list, verbose=True, stop=False):
    """Check error list and stop if one or more are != 0 and stop=True

    Parameters
    ----------
    ex_list: list of integers
        List of exit codes
    verbose: boolean
        Verbose output, default=True
    stop: boolean
        If False (default), does not stop program

    Returns
    -------
    s: integer
        sum of absolute values of exit codes
    """

    if ex_list is None:
        s = 0
    else:
        s = sum([abs(i) for i in ex_list])


    # Evaluate exit codes
    if s > 0:
        n_ex = sum([1 for i in ex_list if i != 0])
        if verbose is True:
            if len(ex_list) == 1:
                print_color('red', 'The last command returned sum|exit codes|={}'.format(s), end='')
            else:
                print_color('red', '{} of the last {} commands returned sum|exit codes|={}'.format(n_ex, len(ex_list), s), end='')
        if stop is True:
            print_color('red', ', stopping')
        else:
            print_color('red', ', continuing')

        if stop is True:
            sys.exit(s)


    return s



def print_color(color, txt, file=sys.stdout, end='\n'):
    """Print text with color. If not supported, print standard text.

    Parameters
    ----------
    color: string
        color name
    txt: string
        message
    file: file handler
        output file handler, default=sys.stdout
    end: string
        end string, default='\n'

    Returns
    -------
    None
    """


    try:
        import colorama
        colors = {'red'    : colorama.Fore.RED,
                  'green'  : colorama.Fore.GREEN,
                  'blue'   : colorama.Fore.BLUE,
                  'yellow' : colorama.Fore.YELLOW,
                  'black'  : colorama.Fore.BLACK,
                 }

        if colors[color] is None:
            col = colorama.Fore.BLACK
        else:
            col = colors[color]

        print(col + txt + colors['black'] + '', file=file, end=end)

    except ImportError:
        print(txt, file=file, end=end)



def substitute(dat, key, val_old, val_new):
    """Performs a substitution val_new for val_old as value corresponding to key.
       See run_csfisher_cut_bins.py

    Parameters
    ----------
    dat: string
        file content
    key: string
        key
    val_old: n/a
        old value
    val_new: n/a
        new value

    Returns
    -------
    dat: string
        file content after substitution
    """

    str_old = '{}\s*=\s*{}'.format(key, val_old)
    str_new = '{}\t\t= {}'.format(key, val_new)

    #print('Replacing \'{}\' -> \'{}\''.format(str_old, str_new))

    dat, n  = re.subn(str_old, str_new, dat)

    if n != 1:
        msg = 'Substitution {} -> {} failed, {} entries replaced'.format(str_old, str_new, n)
        error(msg, val=1)

    return dat



def run_ABC_in_dir(real_dir, n_S, templ_dir):


    files = ['ABC_est_cov.py', 'toy_model_functions.py']
    for f in files:
        copy2('{}/{}'.format(templ_dir, f), '{}/{}'.format(real_dir, f))

    fin  = open('{}/{}'.format(templ_dir, 'toy_model.input'))
    fout = open('{}/{}'.format(real_dir, 'toy_model.input'), 'w')
    dat = fin.read()
    fin.close()

    dat = substitute(dat, 'nsim', 800, n_S)

    fout.write(dat)
    fout.close()

    os.chdir(real_dir)

    run_cmd('python ABC_est_cov.py', run=True, verbose=False)

    os.chdir('../..')



def simulate(n_S_arr, options):

    for i, n_S in enumerate(n_S_arr):

        if options.verbose == True:
            print('{}/{}: n_S={}'.format(i+1, len(n_S_arr), n_S))

        base_dir = 'nsim_{}'.format(n_S)

        # Loop over realisations
        for run in range(options.n_R):

            real_dir = '{}/nr_{}'.format(base_dir, run)
            if not os.path.exists(real_dir):
                os.makedirs(real_dir)

            #if os.path.exists('{}/num_res_nsim_{}.dat'.format(real_dir, n_S)):
            if os.path.exists('{}/num_res.dat'.format(real_dir)):
                 print('Skipping {}'.format(real_dir))
                 #next
            else:
                 print('Running {}'.format(real_dir))
                 run_ABC_in_dir(real_dir, n_S, options.templ_dir)



def read_from_ABC_dirs(n_S_arr, par_name, fit_ABC, options):

    for i, n_S in enumerate(n_S_arr):

        base_dir = 'nsim_{}'.format(n_S)

        for r, run in enumerate(range(options.n_R)):

            real_dir = '{}/nr_{}'.format(base_dir, run)

            #fname = '{}/num_res_nsim_{}.dat'.format(real_dir, n_S)
            fname = '{}/num_res.dat'.format(real_dir)
            if not os.path.exists(fname):
                error('File {} not found'.format(fname))

            fin = open(fname)
            dat = fin.read()
            fin.close()

            for p in par_name:
                for which in ['mean', 'std']:
                    pattern = '{}_{}\s+(\S+)'.format(p, which)
                    res = re.search(pattern, dat)
                    if res is None or len(res.groups()) != 1:
                        error('Entry for {} not found in file {}'.format(pattern, fname))
                    else:
                        val = res.groups()[0]
                        y = getattr(fit_ABC, which)
                        y[p][i][r] = val

    


# Main program
def main(argv=None):
    """Main program.
    """

    par_name = ['a', 'b']            # Parameter list
    delta    = 200

    options = params_default()

    # Number of simulations
    #start = options.n_D + 5
    #stop  = options.n_D * options.f_n_S_max
    start = 58
    stop  = 584
    n_S_arr = np.logspace(np.log10(start), np.log10(stop), options.n_n_S, dtype='int')
    n_n_S = len(n_S_arr)


    # Initialisation of results
    fit_ABC = Results(par_name, n_n_S, options.n_R, file_base='mean_std_ABC', yscale=['linear', 'log'])


    # Create simulations
    if re.search('s', options.mode) is not None:

        simulate(n_S_arr, options)

    # Read simulations
    if re.search('r', options.mode) is not None:

        read_from_ABC_dirs(n_S_arr, par_name, fit_ABC, options)

    fit_ABC.write_mean_std(n_S_arr)

    x1 = np.zeros(shape = options.n_D)
    dpar_exact, det = Fisher_error_ana(x1, options.sig2, delta, mode=-1)

    par = my_string_split(options.spar, num=2, verbose=options.verbose, stop=True)
    options.par = [float(p) for p in par]
    fit_ABC.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact})


    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

