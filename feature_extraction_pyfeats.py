import cv2
import numpy as np
import pyfeats

# Load a 64x64 grayscale image
#image = cv2.imread('../dmg-train/dmg-train/dmg0.txt', cv2.IMREAD_GRAYSCALE)
image = np.loadtxt('../dmg-train/dmg-train/dmg0.txt')

print(image)
# Create a mask that covers the entire image
mask = np.ones(image.shape, dtype=bool)

# 1. First Order Statistics (FOS)
fos_features, fos_labels = pyfeats.fos(image, mask)

# 2. Gray-Level Co-occurrence Matrix (GLCM)
features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(image, ignore_zeros=True)

# 3.1.3 Gray Level Difference Statistics (GLDS)
features, labels = pyfeats.glds_features(f, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])

#3.1.4 Neighborhood Gray Tone Difference Matrix (NGTDM)
features, labels = pyfeats.ngtdm_features(f, mask, d=1)

#3.1.5 Statistical Feature Matrix (SFM)
features, labels = pyfeats.sfm_features(f, mask, Lr=4, Lc=4)

# 3.1.6 Law's Texture Energy Measures (LTE/TEM)
features, labels = pyfeats.lte_measures(f, mask, l=7)

# 3.1.7 Fractal Dimension Texture Analysis (FDTA)
h, labels = pyfeats.fdta(f, mask, s=3)

# 3.1.8 Gray Level Run Length Matrix (GLRLM)
features, labels = pyfeats.glrlm_features(f, mask, Ng=256)

# 3.1.9 Fourier Power Spectrum (FPS)
features, labels = pyfeats.fps(f, mask)

# 3.1.10 Shape Parameters
features, labels = pyfeats.shape_parameters(f, mask, perimeter, pixels_per_mm2=1)

# 3.1.11 Gray Level Size Zone Matrix (GLSZM)
features, labels = pyfeats.glszm_features(f, mask, connectivity=1)

#3.1.12 Higher Order Spectra (HOS)
features, labels = pyfeats.hos_features(f, th=[135,140])

#3.1.13 Local Binary Pattern (LPB)
features, labels = pyfeats.lbp_features(f, mask, P=[8,16,24], R=[1,2,3])

# 3.2.2 Multilevel Binary Morphological Analysis
pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H = pyfeats.multilevel_binary_morphology_features(f, mask, N=30, thresholds=[25, 50]):

#3.3.1 Histogram
H, labels = pyfeats.histogram(f, mask, bins=32)

#3.3.2 Multi-region histogram
features, labels = pyfeats.multiregion_histogram(f, mask, bins, num_eros=3, square_size=3)

#3.3.3 Correlogram
Hd, Ht, labels = pyfeats.correlogram(f, mask, bins_digitize=32, bins_hist=32, flatten=True)

#3.4.1 Fractal Dimension Texture Analysis (FDTA)
h, labels = pyfeats.fdta(f, mask, s=3)

#3.4.2 Amplitude Modulation – Frequency Modulation (AM-FM)
features, labels = pyfeats.amfm_features(f, bins=32)




# Combine all features into a single vector
all_features = np.hstack([
    fos_features, 
    glcm_features, 
    glrlm_features, 
    glszm_features, 
    ngtdm_features, 
    sgldm_features, 
    fd_features, 
    gabor_features
])

# Combine all labels
all_labels = fos_labels + glcm_labels + glrlm_labels + glszm_labels + ngtdm_labels + sgldm_labels + fd_labels + gabor_labels

print(f"Total number of features extracted: {len(all_features)}")
print(f"Feature vector: {all_features}")
print(f"Feature labels: {all_labels}")