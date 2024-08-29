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

for file in os.listdir():
    num_files = len(os.listdir())
    if not file.startswith("mean"):
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
    
print("mean_volume: " , mean_volume)
print("mean_max_x: ", mean_max_x)
print("mean_max_y: ", mean_max_y)
print("mean_min_y: ", mean_min_y)
print("mean_sign_changes: ", mean_sign_changes)
print("mean_max_curvature: ", mean_max_curvature)
print("mean_min_curvature: ", mean_min_curvature)
print("mean_fractal_dimension: ", mean_fractal_dimension)
print("mean_skeleton_lengths: ", mean_skeleton_lengths)

    
