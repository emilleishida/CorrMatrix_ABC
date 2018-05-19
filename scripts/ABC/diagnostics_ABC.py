#!/usr/bin/env python

# diagnostics_ABC.py
# Martin Kilbinger (2017)

# Compability with python2.x for x>6
from __future__ import print_function

import numpy as np
import glob
import sys
import re


def diagnostics_draws():
    """Return diagnostics related to the number of draws.
    """

    files = glob.glob('*')

    Mtot = 0
    nd = 0
    for f in files:

        m = re.findall('linear_PS\d+.dat', f)
        if len(m) == 0:
            continue

        dat = np.genfromtxt(f, names=True, deletechars=['[]'])
        nd += int(dat['NDraws'].sum())
        Mtot += len(dat)

    return Mtot, nd



# Main program
def main(argv=None):
    """Main program.
    """

    Mtot, nd = diagnostics_draws()

    #print('Mtot = {}\t\t\t# Numer of accepted points'.format(Mtot))
    #print('nd   = {}\t\t\t# Number of total draws'.format(nd))
    #print('eta  = {:.4f}\t\t\t# Acceptance rate Mtot/nd'.format(float(Mtot)/float(nd)))
    if nd > 0:
        print('{} {} {:.4f}'.format(Mtot, nd, float(Mtot)/float(nd)))
    else:
        print('- - -')


if __name__ == "__main__":
    sys.exit(main(sys.argv))

