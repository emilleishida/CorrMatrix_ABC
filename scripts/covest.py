import sys
import os
import numpy as np
import errno

import matplotlib
matplotlib.use("Agg")
import pylab as plt

from astropy.table import Table, Column
from astropy.io import ascii



def get_n_S_arr(n_S_min, n_D, f_n_S_max, n_n_S, n_S=None):
    """Return array of values of n_S=number of simulations.

    Parameters
    ----------
    n_S_min: int
        smallest n_S
    n_D: int
        number of data points
    f_n_S_max: float
        largest n_S = f_n_S_max * n_D
    n_n_S: int
        number of values for n_S
    n_S: array of int, optional, default=None
        array of number of simulations, overrides all other arguments if not None

    Returns
    -------
    n_S_arr: array of int
        array with n_S values
    n_n_S: int
        number of n_S values
    """

    if n_S != None:
        n_S_arr = n_S
        n_n_S   = len(n_S_arr)
    else:
        start   = n_S_min
        stop    = n_D * f_n_S_max
        n_S_arr = np.logspace(np.log10(start), np.log10(stop), n_n_S, dtype='int')
 
    return n_S_arr, n_n_S



def no_bias(n, n_D, par):
    """Unbiased estimator of par.
       For example maximum-likelihood estimate of covariance normalised trace,
       TKJ13 (17).
       Or Fisher matrix errors from debiased estimate of inverse covariance.
    """

    return np.asarray([par] * len(n))


def get_n_S_R_from_fit_file(file_base, npar=2):
    """Return array of number of simulations, n_n_S, and number of runs, n_R from fit output file.

    Parameters
    ----------
    file_base: string
        input file name base (without extension)
    npar: int
        number of parameters, default=2

    Returns
    -------
    n_n_S: array of int
        array of number of simulations
    n_R: int
        number of runs
    """

    in_name = '{}.txt'.format(file_base)
    try:
        dat = ascii.read(in_name)
    except IOError as exc:
        if exc.errno == errno.ENOENT:
            error('File {} not found'.format(in_name))
        else:
            raise

    n_n_S = np.array(dat['n_S'].data)
    n_R   = (len(dat.keys()) - 1) / 2 / 2

    return n_n_S, n_R



class param:
    """General class to store (default) variables
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self, **kwds):
        print(self.__dict__)

    def var_list(self, **kwds):
        return vars(self)



class Results:
    """Store results of Fisher matrix and MCMC sampling
    """

    def __init__(self, par_name, n_n_S, n_R, file_base='mean_std', yscale='linear', fct=None):
        """Set arrays for mean and std storing all n_S simulation cases
           with n_R runs each
        """

        self.mean      = {}
        self.std       = {}
        self.par_name  = par_name
        self.file_base = file_base

        if np.isscalar(yscale):
            self.yscale = [yscale, yscale]
        else:
            self.yscale = yscale

        self.fct = fct
        for p in par_name:
            self.mean[p]   = np.zeros(shape = (n_n_S, n_R))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R))

        self.F = np.zeros(shape = (n_n_S, n_R, 2, 2))


    def set(self, par, i, run, which='mean'):
        """Set mean or std for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(self.par_name):
            w = getattr(self, which)
            w[p][i][run] = par[j]


    def get_std_var(self, p):
        """Return standard deviation of the variance over all runs
        """

        n_n_S = self.mean[self.par_name[0]].shape[0]
        std_var = np.zeros(shape = n_n_S)
        for i in range(n_n_S):
            std_var[i] = np.std(self.std[p][i]**2)

        return std_var


    def read_mean_std(self, format='ascii', verbose=False):
        """Read mean and std from file
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            in_name = '{}.txt'.format(self.file_base)
            try:
                dat = ascii.read(in_name)
                for p in self.par_name:
                    for run in range(n_R):
                        col_name = 'mean[{0:s}]_run{1:02d}'.format(p, run)
                        self.mean[p].transpose()[run] = dat[col_name]
                        col_name = 'std[{0:s}]_run{1:02d}'.format(p, run)
                        self.std[p].transpose()[run] = dat[col_name]
            except IOError as exc:
                if exc.errno == errno.ENOENT:
                    if verbose == True:
                        print('File {} not found, continuing'.format(in_name))
                    pass
                else:
                    raise


    def write_mean_std(self, n, format='ascii'):
        """Write mean and std to file
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            cols  = [n]
            names = ['# n_S']
            for p in self.par_name:
                for run in range(n_R):
                    cols.append(self.mean[p].transpose()[run])
                    names.append('mean[{0:s}]_run{1:02d}'.format(p, run))
                    cols.append(self.std[p].transpose()[run])
                    names.append('std[{0:s}]_run{1:02d}'.format(p, run))
            t = Table(cols, names=names)
            f = open('{}.txt'.format(self.file_base), 'w')
            ascii.write(t, f, delimiter='\t')
            f.close()


    def read_Fisher(self, format='ascii'):
        """Read Fisher matrix
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            fname = 'F_{}.txt'.format(self.file_base)
            if os.path.isfile(fname):
                dat = ascii.read(fname)
                for run in range(n_R):
                    for i in (0,1):
                        for j in (0,1):
                            col_name = 'F[{0:d},{1:d}]_run{2:02d}'.format(i, j, run)
                            self.F[:, run, i, j] = dat[col_name].transpose()


    def write_Fisher(self, n, format='ascii'):
        """Write Fisher matrix.
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            cols  = [n]
            names = ['# n_S']
            for run in range(n_R):
                for i in (0,1):
                    for j in (0,1):
                        Fij = self.F[:, run, i, j]
                        cols.append(Fij.transpose())
                        names.append('F[{0:d},{1:d}]_run{2:02d}'.format(i, j, run))
            t = Table(cols, names=names)
            f = open('F_{}.txt'.format(self.file_base), 'w')
            ascii.write(t, f, delimiter='\t')
            f.close()


    def append(self, new, verbose=False):
        """Append new result to self.

        Parameters
        ----------
        self, new: class Result
            previous and new result
        verbose: bool, optional
            verbose mode

        Returns
        -------
        res: bool
            True for success
        """

        n_n_S, n_R         = self.mean[self.par_name[0]].shape
        n_n_S_new, n_R_new = new.mean[new.par_name[0]].shape
        if n_n_S != n_n_S_new:
            error( \
                'Number of simulations different for previous ({}) and new ({}), skipping append...'.format( \
                n_n_S, n_n_S_new), stop=False, verbose=verbose)

            return False

        for p in self.par_name:
            mean      = self.mean[p]
            std       = self.std[p]

            self.mean[p]   = np.zeros(shape = (n_n_S, n_R + n_R_new))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R + n_R_new))
            for n_S in range(n_n_S):
                self.mean[p][n_S]   = np.append(mean[n_S], new.mean[p][n_S])
                self.std[p][n_S]    = np.append(std[n_S], new.std[p][n_S])

        F = self.F
        self.F = np.zeros(shape = (n_n_S, n_R + n_R_new, 2, 2))
        for n_S in range(n_n_S):
            for r in range(n_R):
                self.F[n_S][r] = F[n_S][r]
            for r in range(n_R_new):
                self.F[n_S][n_R + r] = new.F[n_S][r]

        return True


    def plot_mean_std(self, n, n_D, par=None, boxwidth=None, xlog=False):
        """Plot mean and std versus number of realisations n

        Parameters
        ----------
        n: array of integer
            number of realisations {n_S} for ML covariance
        n_D: integer
            dimension of data vector
        par: dictionary of array of float, optional
            input parameter values and errors, default=None
        boxwidth: float, optional
            box width for box plots, default: None, width is determined from n
        xlog: bool, optional
            logarithmic x-axis, default False

        Returns
        -------
        None
        """

        n_R = self.mean[self.par_name[0]].shape[1]

        marker     = ['.', 'D']
        markersize = [6] * len(marker)
        color      = ['b', 'g']

        plot_sth = False
        plot_init(n_D, n_R)

        if boxwidth == None:
            if xlog == False:
                if len(n) > 1:
                    box_width = (n[1] - n[0]) / 2
                else:
                    box_width = 50
            else:
                if len(n) > 1:
                    box_width = np.sqrt(n[1]/n[0]) # ?
                else:
                    box_width = 0.1
            
        else:
            box_width = boxwidth

        if xlog == True:
            fac_xlim = 1.6
            xmin = n[0]/fac_xlim
            xmax = n[-1]*fac_xlim
        else:
            fac_xlim   = 1.05
            xmin = (n[0]-5)/fac_xlim**3
            xmax = n[-1]*fac_xlim

        # Set the number of required subplots (1 or 2)
        j_panel = {}
        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    n_panel = 2
                    j_panel[which] = j+1

        if len(j_panel) == 1:   # Only one plot to do: Use entire canvas
            n_panel = 1
            j_panel[j_panel.keys()[0]] = 1

        if xlog == True:
            width = lambda p, box_width: 10**(np.log10(p)+box_width/2.)-10**(np.log10(p)-box_width/2.)
        else:
            width = lambda p, box_width: np.zeros(len(n)) + float(box_width)

        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    ax = plt.subplot(1, n_panel, j_panel[which])

                    if y.shape[1] > 1:
                        bplot = plt.boxplot(y.transpose(), positions=n, sym='.', widths=width(n, box_width))
                        for key in bplot:
                            plt.setp(bplot[key], color=color[i], linewidth=2)
                        plt.setp(bplot['whiskers'], linestyle='-', linewidth=2)
                    else:
                        plt.plot(n, y.mean(axis=1), marker[i], ms=markersize[i], color=color[i])

                    if xlog == True:
                        ax.set_xscale('log')

                    n_fine = np.arange(xmin, xmax, len(n)/10.0)
                    my_par = par[which]
                    if self.fct is not None and which in self.fct:
                        # Define high-resolution array for smoother lines
                        plt.plot(n_fine, self.fct[which](n_fine, n_D, my_par[i]), '{}-.'.format(color[i]), linewidth=2)

                    plt.plot(n_fine, no_bias(n_fine, n_D, my_par[i]), '{}-'.format(color[i]), label='{}$({})$'.format(which, p), linewidth=2)

        # Finalize plot
        for j, which in enumerate(['mean', 'std']):
            if which in j_panel:
                ax = plt.subplot(1, n_panel, j_panel[which])
                plt.xlabel('$n_{\\rm s}$')
                plt.ylabel('<{}>'.format(which))
                #plt.xticks2()?bo alpha, or n_d / n_s
                plt.xticks(rotation = 'vertical')
                plt.xlim(xmin, xmax)
                ax.set_yscale(self.yscale[j])
                plot_sth = True

                ax.legend(frameon=False)

        if plot_sth == True:
            plt.savefig('{}.pdf'.format(self.file_base))


    def plot_std_var(self, n, n_D, par=None):
        """Plot standard deviation of parameter variance
        """

        n_R = self.mean[self.par_name[0]].shape[1]
        color = ['g', 'm']

        plot_sth = False
        plot_init(n_D, n_R)
        ax = plt.subplot(1, 1, 1)

        for i, p in enumerate(self.par_name):
            y = self.get_std_var(p)
            if y.any():
                plt.plot(n, y, marker='o', color=color[i], label='$\sigma[\sigma^2({})]$'.format(p), linestyle='None')

        for i, p in enumerate(self.par_name):
            y = self.get_std_var(p)
            if y.any():
                if par is not None:
                    n_fine = np.arange(n[0], n[-1], len(n)/10.0)
                    if 'std_var' in self.fct:
                        plot_add_legend(i==0, n_fine, self.fct['std_var'](n_fine, n_D, par[i]), '-', color=color[i], label='This work')
                    if 'std_var_TJK13' in self.fct:
                        plot_add_legend(i==0, n_fine, self.fct['std_var_TJK13'](n_fine, n_D, par[i]), '--', color=color[i], label='TJK13')
                    if 'std_var_TJ14' in self.fct:
                        plot_add_legend(i==0, n_fine, self.fct['std_var_TJ14'](n_fine, n_D, par[i]), '-.', color=color[i], label='TJ14', linewidth=2)

                plt.xlabel('$n_{\\rm s}$')
                plt.ylabel('std(var)')
                plt.legend(loc='best', numpoints=1, frameon=False)
                ax.set_yscale('log')
                plot_sth = True

        plt.ylim(8e-9, 1e-2)

        if plot_sth == True:
            plt.savefig('std_2{}.pdf'.format(self.file_base))



def plot_init(n_D, n_R):

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.16)
    #plt.tight_layout() # makes space for large labels
    ax = plt.gca()

    fs = 16

    ax.yaxis.label.set_size(fs)
    ax.xaxis.label.set_size(fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)

    if n_R>0:
        add_title(n_D, n_R, fs)



def add_title(n_D, n_R, fs):
    """Adds title to plot."""

    plt.suptitle('$n_{{\\rm d}}={}$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R), fontsize=fs)



def plot_add_legend(do_legend, x, y, linestyle, color='b', label='', linewidth=1):

    if do_legend:
        label = label
    else:
        label = None

    plt.plot(x, y, linestyle, color=color, label=label, linewidth=linewidth)



def error(str, val=1, stop=True, verbose=True):
    """Print message str and exits program with code val

    Parameters
    ----------
    str: string
        message
    val: integer
        exit value, default=1
    stop: boolean
        stops program if True (default), continues if False
    verbose: boolean
        verbose output if True (default)

    Returns
    -------
    None
    """

    if verbose is True:
        print>>sys.stderr, "\x1b[31m{}\x1b[0m".format(str),

    if stop is False:
        if verbose is True:
            print>>sys.stderr,  "\x1b[31m{}, continuing...\x1b[0m".format(str),
    else:
        if verbose is True:
            print>>sys.stderr, ''
        sys.exit(val)



def detF(n_D, sig2, delta):
    """Return exact determinant of Fisher matrix.
    """

    det = (n_D/sig2)**2 * delta**2 / 12.0
    return det



def Fisher_error_ana(x, sig2, delta, mode=-1):
    """Return Fisher matrix parameter errors (std), and Fisher matrix detminant, for affine function parameters (a, b)
    """

    n_D = len(x)

    # The four following ways to compute the Fisher matrix errors are statistically equivalent.
    # Note that mode==-1,0 uses the statistical properties mean and variance of the uniform
    # distribution, whereas more=1,2 uses the actual sample x.

    if mode != -1:
        if mode == 2:
            Psi = np.diag([1.0 / sig2 for i in range(n_D)])
            F_11 = np.einsum('i,ij,j', x, Psi, x)
            F_22 = np.einsum('i,ij,j', np.ones(shape=n_D), Psi, np.ones(shape=n_D))
            F_12 = np.einsum('i,ij,j', x, Psi, np.ones(shape=n_D))
        elif mode == 1:
            F_11 = sum(x*x) / sig2
            F_12 = sum(x) / sig2
            F_22 = n_D / sig2
        elif mode == 0:
            F_11 = n_D * delta**2 / 12.0 / sig2
            F_12 = 0
            F_22 = n_D / sig2

        det = F_11 * F_22 - F_12**2
        da2 = F_22 / det
        db2 = F_11 / det
    else:
        # mode=-1
        det = detF(n_D, sig2, delta)
        da2 = 12 * sig2 / (n_D * delta**2)
        db2 = sig2 / n_D

    return np.sqrt([da2, db2]), det



def my_string_split(string, num=-1, verbose=False, stop=False):
    """Split a *string* into a list of strings. Choose as separator
        the first in the list [space, underscore] that occurs in the string.
        (Thus, if both occur, use space.)

    Parameters
    ----------
    string: string
        Input string
    num: int
        Required length of output list of strings, -1 if no requirement.
    verbose: bool
        Verbose output
    stop: bool
        Stop programs with error if True, return None and continues otherwise

    Returns
    -------
    list_str: string, array()
        List of string on success, and None if failed.
    """

    if string is None:
        return None

    has_space      = string.find(' ')
    has_underscore = string.find('_')

    if has_space != -1:
        # string has white-space
        sep = ' '
    else:
        if has_underscore != -1:
        # string has no white-space but underscore
            sep = '_'
        else:
            # string has neither, consists of one element
            if num == -1 or num == 1:
                # one-element string is ok
                sep = None
                pass
            else:
                error('Neither \' \' nor \'_\' found in string \'{}\', cannot split'.format(string))

    #res = string.split(sep=sep) # python v>=3?
    res = string.split(sep)

    if num != -1 and num != len(res):
        if verbose:
            print >>std.styerr, 'String \'{}\' has length {}, required is {}'.format(string, len(res), num)
        if stop is True:
            sys.exit(2)
        else:
            return None

    return res



def log_command(argv, name=None, close_no_return=True):
    """Write command with arguments to a file or stdout.
       Choose name = 'sys.stdout' or 'sys.stderr' for output on sceen.

    Parameters
    ----------
    argv: array of strings
        Command line arguments
    name: string
        Output file name (default: 'log_<command>')
    close_no_return: bool
        If True (default), close log file. If False, keep log file open
        and return file handler

    Returns
    -------
    log: filehandler
        log file handler (if close_no_return is False)
    """

    if name is None:
        name = 'log_' + os.path.basename(argv[0])

    if name == 'sys.stdout':
        f = sys.stdout
    elif name == 'sys.stderr':
        f = sys.stderr
    else:
        f = open(name, 'w')

    for a in argv:

        # Quote argument if special characters
        if ']' in a or ']' in a:
            a = '\"{}\"'.format(a)

        print>>f, a,
        #print>>f, ' ',

    print>>f, ''

    if close_no_return == False:
        return f

    if name != 'sys.stdout' and name != 'sys.stderr':
        f.close()




