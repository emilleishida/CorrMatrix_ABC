#!/usr/bin/env python

"""
:Name: nicaea_ABC.py

:Description: Python interface to nicaea.
Functions are copied from ~/python/nicaea.py

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

:Date: 2017

:Package: ABC
"""


import covest

from astropy.io import ascii


def run_nicaea(path, lmin, lmax, nell, par_name=None, par_val=None):
    """Calls nicaea.

    Parameters
    ----------
    path: string
        path to nicaea
    lmin: double
        minimum ell
    lmax: double
        maximume ell
    nell: int
        number of ell modes
    par_name: array of string, optional, default=None
        parameter names for on-the-fly updates
    par_val: array of float, optiona, default=None
	parameter values corresponding to par_nam 

    Returns
    -------
    None
    """

    Lstr = '-L \'{} {} {}\''.format(lmin, lmax, nell)
    parstr = '' 
    if par_name is not None:
        for i, name in enumerate(par_name):
            parstr = '{} --{} {}'.format(parstr, name, par_val[i])
    err = covest.run_cmd('{}/bin/lensingdemo -D 0 {} {} -H 1'.format(path, Lstr, parstr), verbose=True, stop=False)

    if err != 0:
        print('Nicaea returned with error code {}'.format(err))

    return err



def read_Cl(path, fname):
    """Read and return theoretical power spectrum.

    Parameters
    ----------
    path: string
        path to nicaea output file
    fname: string
        file name

    Returns
    -------
    ell: array of double
        angular Fourier modes
    C_ell: array of double
        power spectrum
    """

    dat   = ascii.read('{}/{}'.format(path, fname))
    ell   = dat['l'].data
    c_ell = dat['P_k^00(l)'].data

    return ell, c_ell



def create_link(dst, path_cosmo):
    """Create link to file 'path_cosmo/dst'.

    Parameters
    ----------

    dst: string
        Destination file name
    path_cosmo: string
        Directory name

    Returns
    -------
    None
    """

    if os.path.isfile(dst):
        pass
    else:
        src = '{}/{}'.format(path_cosmo, dst)
        if os.path.isfile(src):
            os.symlink(src, dst)
        else:
            stuff.error('File {} not found at dir {}'.format(dst, path_cosmo))



def create_links_to_cosmo(path_to_cosmo):
    """Create links to all cosmo files.

    Parameters
    ----------
    path_to_cosmo: string
        Directory name

    Returns
    -------
    None
    """

    files = ['cosmo.par', 'cosmo_lens.par', 'nofz.par']
    for dst in files:
        create_link(dst, path_cosmo)

    files = glob.glob('{}/nofz_*'.format(path_cosmo))
    for dst in files:
        create_link(os.path.basename(dst), path_cosmo)


