#!/usr/bin/env python

# Compability with python2.x for x>6
from __future__ import print_function

# job_ABC.py
# Martin Kilbinger (2017)


import sys
import os
import re
import subprocess
import copy

import shlex
from shutil import copy2

from optparse import OptionParser
from optparse import OptionGroup

import numpy as np

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

    n_D = 750

    p_def = param(
        n_D = n_D,
        n_R = 4,
        n_n_S = 10,
        f_n_S_max = 10.0,
        n_S_min = n_D + 5,
        spar = '1.0 0.0',
        spar_name = 'a_b',
        sig2 = 5.0,
        model = 'affine',
        verbose = True,
        templ_dir = './templates',
        mode = 's',
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

    parser.add_option('-D', '--n_D', dest='n_D', type='int', default=p_def.n_D,
        help='Number of data points, default={}'.format(p_def.n_D))
    parser.add_option('-R', '--n_R', dest='n_R', type='int', default=p_def.n_R,
        help='Number of runs per simulation, default={}'.format(p_def.n_R))
    parser.add_option('', '--n_n_S', dest='n_n_S', type='int', default=p_def.n_n_S,
        help='Number of n_S, where n_S is the number of simulation, default={}'.format(p_def.n_n_S))
    parser.add_option('', '--f_n_S_max', dest='f_n_S_max', type='float', default=p_def.f_n_S_max,
        help='Maximum n_S = n_D x f_n_S_max, default: f_n_S_max={}'.format(p_def.f_n_S_max))
    parser.add_option('', '--n_S_min', dest='n_S_min', type='int', default=p_def.n_S_min,
        help='Minimum n_S, default=n_D+5 ({})'.format(p_def.n_S_min))
    parser.add_option('', '--n_S', dest='str_n_S', type='string', default=None,
        help='Array of n_S, default=None. If given, overrides n_S_min, n_n_S and f_n_S_max')

    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='Parameter array, for plotting, default=\'{}\''.format(p_def.spar))
    parser.add_option('-P', '--par_name', dest='spar_name', type='string', default=p_def.spar_name,
        help='Parameter names, default=\'{}\''.format(p_def.spar_name))
    parser.add_option('-M', '--model', dest='model', type='string', default=p_def.model,
        help='Model, one in \'affine\', \'quadratic\', default=\'{}\''.format(p_def.model))

    parser.add_option('-m', '--mode', dest='mode', type='string', default=p_def.mode,
        help='Mode: \'s\'=simulate, \'r\'=read ABC dirs, \'R\'=read master file, default={}'.format(p_def.mode))
    parser.add_option('', '--recov_iter', dest='recov_iter', action='store_true',
        help='Re-estimate covariance at beginning ofevery iteration')

    parser.add_option('-b', '--boxwidth', dest='boxwidth', type='float', default=None,
        help='box width for box plot, default=None, determined from n_S array')
    parser.add_option('', '--xlog', dest='xlog', action='store_true',
        help='logarithmic x-axis for box plots')

    parser.add_option('', '--template_dir', dest='templ_dir', type='string', default=p_def.templ_dir,
        help='Template directory, default=\'{}\''.format(p_def.templ_dir))

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

    if options.str_n_S == None:
        param.n_S = None
    else:
        str_n_S_list = my_string_split(options.str_n_S, verbose=False, stop=True)
        param.n_S = [int(str_n_S) for str_n_S in str_n_S_list]

    param.par_name = my_string_split(options.spar_name, verbose=False, stop=True)

    return param



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



def run_ABC_in_dir(real_dir, n_S, templ_dir, nruns=-1, prev_run=-1):
    """ Runs or continues ABC in given directory.

    Parameters
    ----------
    real_dir: string
        target directory
    n_S: int
        number of simulations for covariance estimation
    templ_dir: string
        template directory
    nruns: int, optional, default=-1
        maximum number of runs (if < 0 run until convergence)
    prev_run: int, optional, default=-1
        if > 0, continue ABC from iteration #prev_run instead of running from start

    Returns
    -------
    None
    """

    files = ['ABC_est_cov.py', 'toy_model_functions.py']
    files_opt = ['cov_SSC_rel_log.txt', 'cov_SSC_rel_lin.txt']

    for f in files:
        copy2('{}/{}'.format(templ_dir, f), '{}/{}'.format(real_dir, f))
    for f in files_opt:
        source = '{}/{}'.format(templ_dir, f)
        if os.path.exists(source):
            copy2(source, '{}/{}'.format(real_dir, f))

    fin  = open('{}/{}'.format(templ_dir, 'toy_model.input'))
    fout = open('{}/{}'.format(real_dir, 'toy_model.input'), 'w')
    dat = fin.read()
    fin.close()

    dat = substitute(dat, 'nsim', 800, n_S)
    dat = substitute(dat, 'nruns', -1, nruns)

    if prev_run > 0:
        # Continue ABC from previous run, need observation from file
        dat = substitute(dat, 'path_to_obs', 'None', 'observation_xy.txt')

    fout.write(dat)
    fout.close()

    os.chdir(real_dir)

    if prev_run < 0:
        run_cmd('python ABC_est_cov.py', run=True, verbose=False)
    else:
        # Re-compute and write estimated coariance
        run_cmd('python ABC_est_cov.py --only_cov_est', run=True, verbose=False)
        # Continue ABC for one run
        run_cmd('continue_ABC.py -i toy_model.input -f toy_model_functions.py -p {}'.format(prev_run), run=True, verbose=False)

    os.chdir('../..')



def simulate(n_S_arr, param):

    if param.verbose == True:
        print('Creating {} simulations with {} runs each'.format(len(n_S_arr), param.n_R))

    for i, n_S in enumerate(n_S_arr):

        if param.verbose == True:
            print('{}/{}: n_S={}'.format(i+1, len(n_S_arr), n_S))

        base_dir = 'nsim_{}'.format(n_S)

        # Loop over realisations
        for run in range(param.n_R):

            real_dir = '{}/nr_{}'.format(base_dir, run)
            if not os.path.exists(real_dir):
                os.makedirs(real_dir)

            if os.path.exists('{}/num_res.dat'.format(real_dir)):
                if param.verbose == True:
                    print('Skipping {}'.format(real_dir))
            else:
                if param.verbose == True:
                    print('Running {}'.format(real_dir))

                if not param.recov_iter:

                    run_ABC_in_dir(real_dir, n_S, param.templ_dir)

                else:

                    # First time: Run main script ABC_est_cov.py for one iteration
                    run_ABC_in_dir(real_dir, n_S, param.templ_dir, nruns=1)

                    # This should be a bit larger than the max iteration expected
                    nruns_max = 30

                    # Loop over calls of continue_ABC.py, each time running for nruns=1 
                    for prev_run in range(1, nruns_max):
                        run_ABC_in_dir(real_dir, n_S, param.templ_dir, nruns=1, prev_run=prev_run)



def read_from_ABC_dirs(n_S_arr, par_name, fit_ABC, options):

    if options.verbose == True:
        print('Reading simulation results (mean, std) from disk (ABC run directories)')

    for i, n_S in enumerate(n_S_arr):

        base_dir = 'nsim_{}'.format(n_S)

        for r, run in enumerate(range(options.n_R)):

            real_dir = '{}/nr_{}'.format(base_dir, run)

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



def Fisher_ana_quad_read_par(templ_dir, par, mode=1):
    """Read parameters from config file and return Fisher matrix errors
       on parameters of quadratic model.
    """

    from cosmoabc.ABC_functions import read_input

    filename = '{}/{}'.format(templ_dir, 'toy_model.input')
    Parameters = read_input(filename)

    f_sky = float(Parameters['f_sky'][0])
    sigma_eps = float(Parameters['sigma_eps'][0])
    nbar      = float(Parameters['nbar'][0])
    nbar_amin2 = units.Unit('{}/arcmin**2'.format(nbar))
    nbar_rad2  = nbar_amin2.to('1/rad**2')
    
    nell      = int(Parameters['nell'][0])
    logellmin = float(Parameters['logellmin'][0])
    logellmax = float(Parameters['logellmax'][0])
    ellmode = Parameters['ellmode'][0]
    if ellmode == 'log':
        # Equidistant in log ell
        logell = np.linspace(logellmin, logellmax, nell)
    else:
        # Equidistant in ell
        ellmin = 10**logellmin
        ellmax = 10**logellmax
        ell = np.linspace(ellmin, ellmax, nell)
        logell = np.log10(ell)

    cov_model = Parameters['cov_model'][0]
    print('Covariance used in Fisher matrix = {}'.format(cov_model))

    ampl_fid, tilt_fid = par

    dpar, det = Fisher_ana_quad(10**logell, f_sky, sigma_eps, nbar_rad2, ampl_fid, tilt_fid, cov_model,
                                ellmode=ellmode, mode=mode, templ_dir=templ_dir)
    return dpar, det, nell
    


# Main program
def main(argv=None):
    """Main program.
    """

    p_def = params_default()
    options, args = parse_options(p_def)

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)

    # Save calling command
    log_command(argv)
    if options.verbose:
        log_command(argv, name='sys.stderr')


    # Number of simulations
    n_S_arr, n_n_S = get_n_S_arr(param.n_S_min, param.n_D, param.f_n_S_max, param.n_n_S, n_S=param.n_S)


    # Display n_S array and exit
    if re.search('d', param.mode) is not None:
        print('n_S =', n_S_arr)
        return 0

    # Initialisation of results
    fit_ABC = Results(param.par_name, n_n_S, param.n_R, file_base='mean_std_ABC', yscale=['linear', 'log'], fct={})


    # MKDEBUG TODO add check: n_D and nobs in toy_model.input should be consistent, if only for plotting reasons

    # Create simulations
    if re.search('s', param.mode) is not None:

        simulate(n_S_arr, param)

    # Read simulations from ABC run directories and write to master file
    if re.search('r', param.mode) is not None:

        read_from_ABC_dirs(n_S_arr, param.par_name, fit_ABC, param)
        fit_ABC.write_mean_std(n_S_arr)

    if re.search('R', param.mode) is not None:

        if param.verbose == True:
            print('Reading simulation results (mean, std) from disk (master file)')
        fit_ABC.read_mean_std()

    par = my_string_split(param.spar, num=2, verbose=param.verbose, stop=True)
    param.par = [float(p) for p in par]

    if param.model == 'affine':
        x1 = np.zeros(shape = param.n_D) # Dummy variable
        delta = 200
        dpar_exact, det = Fisher_error_ana(x1, param.sig2, delta, mode=-1)
        n_D  = param.n_D
    elif param.model == 'quadratic':
        for mode in ([0, 1]):
            dpar_exact, det, n_D = Fisher_ana_quad_read_par(param.templ_dir, param.par, mode=mode)
            print('input par and exact std:          ', end='')
            for i, p in enumerate(param.par):
                print('{:.4f}  +- {:.5f}             '.format(p, dpar_exact[i]), end='')
            print('')

            print('estim mean std(mean) [mean(std)]: ', end='')
        std_estim = []
        for p in param.par_name:
            mean, std, std2 = fit_ABC.get_mean_std_all(p)
            std_estim.append(std)
            print('{:.5f} +- {:.5f} [{:.5f}]'.format(mean, std, std2), end='   ')
        print('')
    else:
        raise ABCCovError('Unknown model \'{}\''.format(param.model))

    try:
        fit_ABC.plot_mean_std(n_S_arr, n_D, par={'mean': param.par, 'std': dpar_exact}, boxwidth=param.boxwidth, xlog=param.xlog, model=param.model)
        #fit_ABC.plot_mean_std(n_S_arr, n_D, par={'mean': param.par, 'std': std_estim}, boxwidth=param.boxwidth, xlog=param.xlog, model=param.model)
        dpar2 = dpar_exact**2
        fit_ABC.plot_std_var(n_S_arr, n_D, par=dpar2, xlog=param.xlog)
    except:
        print('Error occured while plotting ABC mean and std. Maybe just the display could not be accessed. Continuing anyway...')
        pass


    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))


