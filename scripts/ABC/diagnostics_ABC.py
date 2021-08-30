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
    nit = 0
    for f in files:

        #m = re.findall('linear_PS\d+.dat', f)
        m = re.findall('quad_PS\d+.dat', f)
        if len(m) == 0:
            continue

        dat = np.genfromtxt(f, names=True, deletechars=['[]'])
        nd += int(dat['NDraws'].sum())
        Mtot += len(dat)
        nit += 1

    return Mtot, nd, nit



# Main program
def main(argv=None):
    """Main program.
    """

    Mtot, nd, nit = diagnostics_draws()

    print('# Mtot nd ratio nit')
    if nd > 0:
        print('{} {} {:.4f} {}'.format(Mtot, nd, float(Mtot)/float(nd), nit))
    else:
        print('- - - -')


if __name__ == "__main__":
    sys.exit(main(sys.argv))

