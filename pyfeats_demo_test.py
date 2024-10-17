import numpy as np
from pyfeats import *
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import os

def extract_features(file):
    filepath = '../mat-dist-test/mat-dist-test'
    image = np.loadtxt(filepath + '/' + file)

    save_path = '../dmg-test/features'

    if os.path.isfile(save_path + '/' + 'feature'+file):
        return
    
    print(file)

    # Create a mask that covers the entire image
    points = np.zeros(image.shape, dtype=np.int32)
    points[0,0] = 1
    points[0,63] = 1
    points[63,0] = 1
    points[63,63] = 1
    #points.reshape((-1,1,2))
    mask = np.ones(image.shape, dtype=bool)

    #%% Pre-processing
    def padImage(im, pad=2, value=0):
        TDLU=[1, 1, 1, 1]  #top, down, left, right pad
        out = im.copy()
        for _ in range(pad):
            out = cv2.copyMakeBorder(out, TDLU[0], TDLU[1], TDLU[2], TDLU[3],\
                                        cv2.BORDER_CONSTANT, None, value)
        return out

    # Get the coordinates of the points
    coordinates = np.array([[0, 0], [0, 63], [63, 0], [63, 63]], dtype=np.int32)

    # Draw the perimeter
    perimeter = cv2.polylines(np.zeros(image.shape, dtype=np.uint8), [coordinates], isClosed=True, color=1, thickness=1).astype('i')

        
    ## Fill image with zeros outside ROI
    #image = np.multiply(image.astype(np.double), mask).astype(np.uint8)
    #max_point_x = max(points[:,0])
    #max_point_y = max(points[:,1])
    #min_point_x = min(points[:,0])
    #min_point_y = min(points[:,1])  
    #a,b,c,d = min_point_y, (max_point_y+1), min_point_x, (max_point_x+1)
    #image, mask, perimeter = image[a:b,c:d], mask[a:b,c:d], perimeter[a:b,c:d]
    #image, mask, perimeter = padImage(image, 2, 0), padImage(mask, 2, 0), padImage(perimeter, 2, 0) # ALWAYS PAD IMAGE SO AS SHAPE PARAMETERS WORK
    #
    #plt.imshow(image, cmap='gray')
    #plt.title('Image cropped to ROI')
    #plt.show()
    #
    #plt.imshow(mask, cmap='gray')
    #plt.title('Mask of image cropped to ROI')
    #plt.show()
    #
    #plt.imshow(perimeter, cmap='gray')
    #plt.title('Perimeter of image cropped to ROI')
    #plt.show()

    #%% A1. Texture features
    features = {}
    features['A_FOS'] = fos(image, mask)
    features['A_GLCM'] = glcm_features(image, ignore_zeros=False)
    features['A_GLDS'] = glds_features(image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    features['A_NGTDM'] = ngtdm_features(image, mask, d=1)
    features['A_SFM'] = sfm_features(image, mask, Lr=4, Lc=4)
    features['A_LTE'] = lte_measures(image, mask, l=7)
    features['A_FDTA'] = fdta(image, mask, s=3)
    features['A_GLRLM'] = glrlm_features(image, mask, Ng=256)
    features['A_FPS'] = fps(image, mask)
    features['A_Shape_Parameters'] = shape_parameters(image, mask, perimeter, pixels_per_mm2=1)
    features['A_HOS'] = hos_features(image, th=[135,140])
    features['A_LBP'] = lbp_features(image, image, P=[8,16,24], R=[1,2,3])
    features['A_GLSZM'] = glszm_features(image, mask)
    #plot_sinogram(image, image_name)

    #%% B. Morphological features
    features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'] = grayscale_morphology_features(image, N=30)
    features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], \
    features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'] = multilevel_binary_morphology_features(image, mask, N=30, thresholds=[25,50])

    #plot_pdf_cdf(features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'], image_name)
    #plot_pdfs_cdfs(features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'])

    #%% C. Histogram Based features
    features['C_Histogram'] = histogram(image, mask, bins=32)
    features['C_MultiregionHistogram'] = multiregion_histogram(image, mask, bins=32, num_eros=3, square_size=3)
    #features['C_Correlogram'] = correlogram(image, mask, bins_digitize=32, bins_hist=32, flatten=True)

    #plot_histogram(image, mask, bins=32, name=image_name)
    #plot_correlogram(image, mask, bins_digitize=32, bins_hist=32, name=image_name)

    #%% D. Multi-Scale features
    features['D_DWT'] = dwt_features(image, mask, wavelet='bior3.3', levels=3)
    features['D_SWT'] = swt_features(image, mask, wavelet='bior3.3', levels=3)
    features['D_WP'] = wp_features(image, mask, wavelet='coif1', maxlevel=3)
    features['D_GT'] = gt_features(image, mask)
    features['D_AMFM'] = amfm_features(image)

    #%% E. Other
    features['E_HOG'] = hog_features(image, ppc=8, cpb=3)
    features['E_HuMoments'] = hu_moments(image)
    #features['E_TAS'] = tas_features(image)
    features['E_ZernikesMoments'] = zernikes_moments(image, radius=9)

    ##%% Print
    #for x, y in features.items():
    #    if x.startswith("B") | x.startswith("C") | x.startswith("E"):
    #        continue
    #    print('-' * 50)
    #    print(x)
    #    print('-' * 50)
    #    if len(y[1]) == 1:
    #        print(y[1][0], '=', y[0])
    #    else:
    #        for i in range(len(y[1])):
    #        print(y[1][i], '=', y[0][i])

    features_labels = []
    features_values = []
    for x, y in features.items():
        if x.startswith("B") | x.startswith("C") | x.startswith("E"):
            continue
        #print('-' * 50)
        #print(x)
        #print('-' * 50)
        if len(y[1]) == 1:
            #print(y[1][0], '=', y[0])
            features_labels.append(y[1][0])
            features_values.append(y[0])
        else:
            for i in range(len(y[1])):
                #print(y[1][i], '=', y[0][i])
                features_labels.append(y[1][i])
                features_values.append(y[0][i])
    
    np.savetxt(save_path + '/' + 'feature'+file,features_values)
    #return features_values, features_labels


# extract features for test and training data
filepath = '../mat-dist-test/mat-dist-test'
files = os.listdir(filepath)

#save_path = '../dmg-test/features'

with multiprocessing.Pool(processes=33) as pool:
    results = pool.map(extract_features, files)

#for file in files:
#    #print(file)
#    if str(file).startswith('mat'):
#        #print("HI")
#        extract_features(file)