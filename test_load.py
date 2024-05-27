from numpy import loadtxt
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt

#with io.open("../dmg-train/dmg-train/dmg0.txt", 'r+') as input:

crack = loadtxt("../dmg-train/dmg-train/dmg0.txt")

skeleton = skeletonize(crack)

fig,axes = plt.subplots(nrows = 1, ncols = 2)

ax = axes.ravel()

ax[0].imshow(crack,cmap = plt.cm.gray)

ax[1].imshow(skeleton,cmap = plt.cm.gray)

fig.savefig("test.png")



