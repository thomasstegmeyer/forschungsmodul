#this is a file to extract all the relevant measurements from the dmg files

import os
import numpy as np

mean_volume = 0.0
mean_max_x = 0.0
mean_max_y = 0.0
mean_min_y = 0.0
mean_sign_changes = 0.0
mean_max_curvature = 0.0
mean_min_curvature = 0.0
mean_fractal_dimension = 0.0
mean_skeleton_lengths = 0.0

std_volume = 0.0
std_max_x = 0.0
std_max_y = 0.0
std_min_y = 0.0
std_sign_changes = 0.0
std_max_curvature = 0.0
std_min_curvature = 0.0
std_fractal_dimension = 0.0
std_skeleton_lengths = 0.0

for file in os.listdir():
    num_files = len(os.listdir())
    if not file.startswith("normalize"):
        measures = np.loadtxt(file)
        mean_volume += measures[0]/num_files
        mean_max_x += measures[1]/num_files
        mean_max_y += measures[2]/num_files
        mean_min_y += measures[3]/num_files
        mean_sign_changes += measures[4]/num_files
        mean_max_curvature += measures[5]/num_files
        mean_min_curvature += measures[6]/num_files
        mean_fractal_dimension += measures[7]/num_files
        mean_skeleton_lengths += measures[8]/num_files

for file in os.listdir():
    num_files = len(os.listdir())
    if not file.startswith("normalize"):
        measures = np.loadtxt(file)
        std_volume += (measures[0]-mean_volume)**2/num_files
        std_max_x += (measures[1]-mean_max_x)**2/num_files
        std_max_y += (measures[2]-mean_max_y)**2/num_files
        std_min_y += (measures[3]-mean_min_y)**2/num_files
        std_sign_changes += (measures[4]-mean_sign_changes)**2/num_files
        std_max_curvature += (measures[5]-mean_max_curvature)**2/num_files
        std_min_curvature += (measures[6]-mean_min_curvature)**2/num_files
        std_fractal_dimension += (measures[7]-mean_fractal_dimension)**2/num_files
        std_skeleton_lengths += (measures[8]-mean_skeleton_lengths)**2/num_files

std_volume = np.sqrt(std_volume)
std_max_x = np.sqrt(std_max_x)
std_max_y = np.sqrt(std_max_y)
std_min_y = np.sqrt(std_min_y)
std_sign_changes = np.sqrt(std_sign_changes)
std_max_curvature = np.sqrt(std_max_curvature)
std_min_curvature = np.sqrt(std_min_curvature)
std_fractal_dimension = np.sqrt(std_fractal_dimension)
std_skeleton_lengths = np.sqrt(std_skeleton_lengths)

np.savetxt("../means.txt",[mean_volume,mean_max_x,mean_max_y,mean_min_y,mean_sign_changes,mean_max_curvature,mean_min_curvature,mean_fractal_dimension,mean_skeleton_lengths])
np.savetxt("../std.txt",[std_volume,std_max_x,std_max_y,std_min_y,std_sign_changes,std_max_curvature,std_min_curvature,std_fractal_dimension,std_skeleton_lengths])







#for file in os.listdir():
#    if not file.startswith("normalize"):
#        measures = np.loadtxt(file)
#        
#
#        norm_volume = (measures[0]-mean_volume)/std_volume
#        norm_max_x = (measures[1]-mean_max_x)/std_max_x
#        norm_max_y = (measures[2]-mean_max_y)/std_max_y
#        norm_min_y = (measures[3]-mean_min_y)/std_min_y
#        norm_sign_changes = (measures[4]-mean_sign_changes)/std_sign_changes
#        norm_max_curvature = (measures[5]-mean_max_curvature)/std_max_curvature
#        norm_min_curvature = (measures[6]-mean_min_curvature)/std_min_curvature
#        norm_fractal_dimension = (measures[7]-mean_fractal_dimension)/std_fractal_dimension
#        norm_skeleton_lengths = (measures[8]-mean_skeleton_lengths)/std_skeleton_lengths
#
#        path = "../measures_normalized/" + file
#        np.savetxt(path,[norm_volume,norm_max_x,norm_max_y,norm_min_y,norm_sign_changes,norm_max_curvature,norm_min_curvature,norm_fractal_dimension,norm_skeleton_lengths])
#
#
#[norm_volume,norm_max_x,norm_max_y,norm_min_y,norm_sign_changes,norm_max_curvature,norm_min_curvature,norm_fractal_dimension,norm_skeleton_lengths]
#
#print("mean_volume: " , mean_volume)
#print("mean_max_x: ", mean_max_x)
#print("mean_max_y: ", mean_max_y)
#print("mean_min_y: ", mean_min_y)
#print("mean_sign_changes: ", mean_sign_changes)
#print("mean_max_curvature: ", mean_max_curvature)
#print("mean_min_curvature: ", mean_min_curvature)
#print("mean_fractal_dimension: ", mean_fractal_dimension)
#print("mean_skeleton_lengths: ", mean_skeleton_lengths)
#
    
