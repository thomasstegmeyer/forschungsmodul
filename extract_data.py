#this is a file to extract all the relevant measurements from the dmg files

#needs to be in dmg folder to work

import os
import numpy as np
import skimage
import astropy.units as u
from fil_finder import FilFinder2D
import porespy as ps
from scipy import ndimage

def edge_lengths(image: np.ndarray):
    """Calculate the length of the edges in a binary image."""

    kernel = np.array([[2, 1, 2], [1, 0, 1], [2, 1, 2]])
    neighbour_count = ndimage.convolve(
        image.astype(np.uint8), kernel, mode="constant", cval=0
    )
    straight = (neighbour_count == 2) & image
    diagonal = (neighbour_count == 4) & image
    half_diag = (neighbour_count == 3) & image

    # Count the number of diagonal and straight pixels
    straight_length = np.sum(straight)
    diagonal_length = np.sum(diagonal) * np.sqrt(2)
    half_diag_length = np.sum(half_diag) * (1 + np.sqrt(2) / 2)

    # Add them together to get the total length
    edge_length = straight_length + diagonal_length + half_diag_length
    return edge_length



    
def calc_local_curvature(curve, stride=1) -> np.ndarray:
    """Calc the curvature at every point of the curve"""
    dx = np.gradient(curve[::stride, 0])
    dy = np.gradient(curve[::stride, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    nominator = dx * d2y - dy * d2x
    denominator = dx**2 + dy**2 + 1e-11
    curvature = nominator / denominator**1.5
    return curvature


def max_index(arr):
    i = len(arr)-1
    while arr[i] == 0 and i > 0:
        i = i-1
    return i

def min_index(arr):
    i = 0
    while arr[i] == 0 and i < len(arr)-1:
        i = i+1
    return i

def nearest_neighbor(point,tuple_list):
    distances = []
    for tuple in tuple_list:
        distances.append((point[0]-tuple[0])**2+(point[1]-tuple[1])**2)
    idx = distances.index(min(distances))
    return tuple_list[idx]

def skeleton_array_to_coordinates(skeleton_array):
    #print("skeleton array")
    #print(skeleton_array)
    #get all nonzero coordinates
    skeleton_coordinates = list(map(tuple,np.transpose(np.nonzero(np.transpose(skeleton_array)))))
    #skeleton_coordinates = list(map(tuple,np.nonzero(skeleton_array)))
    skeleton_coordinates_in_order = []
    #get starting coordinate
    for i in range(len(skeleton_coordinates)):
        if not skeleton_coordinates[i][1] == 0:
            skeleton_coordinates_in_order.append(skeleton_coordinates[i])
            skeleton_coordinates.remove(skeleton_coordinates[i])
            break
    #print(skeleton_coordinates_in_order)
        
    #iterate
    
    while len(skeleton_coordinates)>0:
        #call function for nearest neighbor
        neighbor = nearest_neighbor(skeleton_coordinates_in_order[-1],skeleton_coordinates)
        skeleton_coordinates_in_order.append(neighbor)
        #delete current point from array
        skeleton_coordinates.remove(neighbor)
    

    #return coordinates
    return skeleton_coordinates_in_order


def extract_data(damage_array):
    skeleton = skimage.morphology.skeletonize(damage_array)
    #prune skeleton and calculate curvature:
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.create_mask(border_masking=True, verbose=False,use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    coord = skeleton_array_to_coordinates(fil.skeleton)
    coord = np.array(coord)
    curvatures = calc_local_curvature(coord)

    # --------------- important measures -----------------------------
    measurements = []
    #volume
    measurements.append(np.sum(damage_array))
    #max x
    measurements.append(max_index(np.sum(skeleton,axis = 0)))
    #max y
    measurements.append(max_index(np.sum(skeleton,axis = 1)))
    #min y 
    measurements.append(min_index(np.sum(skeleton,axis = 1)))
    
    #sign changes
    measurements.append(np.sum(np.diff(np.sign(curvatures))!=0))
    #max curvature
    measurements.append(max(curvatures))
    #min curvature
    measurements.append(min(curvatures))

    #fractal dimension
    sizes, counts, slopes = ps.metrics.boxcount(damage_array)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    ps_dim = -coeffs[0]
    measurements.append(ps_dim)

    #skeleton lengths
    measurements.append(edge_lengths(skeleton))


    return measurements

i = 0
for file in os.listdir():
    damage = np.loadtxt(file)
    measures = extract_data(damage)
    name = "measures/measure" + str(i) + ".txt"
    np.savetxt(name,measures)
    i+=1