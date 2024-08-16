import matplotlib.pyplot as plt
import astropy.units as u

import matplotlib as mpl

from fil_finder import FilFinder2D
from astropy.io import fits

from numpy import loadtxt

crack = loadtxt("../dmg-train/dmg-train/dmg0.txt")

fil = FilFinder2D(crack, ang_scale=0.1 * u.deg)

fil.medskel(verbose=True)
print("here")
print(fil.analyze_skeletons())
print(fil.lengths())

plt.savefig('test_fil.png',fil.skeleton, origin='lower')
