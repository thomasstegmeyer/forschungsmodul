import os
import numpy as np

folders = os.listdir("../ordered")

### single mean predictor
means = np.loadtxt("means.txt")
#print(means)

rel_errors = []

for folder in folders:
    measures = os.listdir("../ordered/"+folder+"/measures")
    rel_errors_per_folder = []
    for measure in measures:
        loaded_measure = np.loadtxt("../ordered/"+folder+"/measures/"+measure)
        rel_error = np.abs((loaded_measure-means)/loaded_measure)
        rel_errors.append(rel_error)
        rel_errors_per_folder.append(rel_error)

    rel_errors_folder_average = np.mean(rel_errors_per_folder,axis = 0)
    print(str(folder) + ": " + str(rel_errors_folder_average))

rel_errors_average = np.mean(rel_errors, axis = 0)

print("relative error complete mean: " + str(rel_errors_average))





### different means per class

rel_errors = []


for folder in folders:
    means = np.loadtxt("../ordered/"+folder+"/means.txt")
    #print(means)
    measures = os.listdir("../ordered/"+folder+"/measures")
    rel_errors_per_folder = []
    for measure in measures:
        loaded_measure = np.loadtxt("../ordered/"+folder+"/measures/"+measure)
        rel_error = np.abs((loaded_measure-means)/loaded_measure)
        rel_errors.append(rel_error)
        rel_errors_per_folder.append(rel_error)
    rel_errors_folder_average = np.mean(rel_errors_per_folder,axis = 0)
    print(str(folder) + ": " + str(rel_errors_folder_average))

rel_errors_average = np.mean(rel_errors, axis = 0)

print("relative error different means: " + str(rel_errors_average))