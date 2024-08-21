import os
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import shutil
import numpy as np
import time

# Paths to the folders
folder1 = '../mat-dist-test/mat-dist-test'  # Replace with your first folder path
folder2 = '../dmg-train/measures' # Replace with your second folder path

# Output folders
output_shoe = '../ordered/shoe'
output_top = '../ordered/top'
output_bottom = '../ordered/bottom'
output_bag = '../ordered/bag'
output_dress = '../ordered/dress'
output_garbage = '../ordered/garbage'

# Make sure output folders exist
os.makedirs(output_shoe, exist_ok=True)
os.makedirs(output_top, exist_ok=True)
os.makedirs(output_bottom, exist_ok=True)
os.makedirs(output_bag, exist_ok=True)
os.makedirs(output_dress, exist_ok=True)
os.makedirs(output_garbage, exist_ok=True)

# Function to extract the number from the filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group(0) if match else None



# Process each file in the first folder
for filename1 in os.listdir(folder1):
    if filename1.endswith('.txt'):
        number = extract_number(filename1)
        if number is None:
            print(f"Could not extract a number from {filename1}, skipping.")
            continue

        # Find the corresponding file in the second folder
        corresponding_file2 = None
        for filename2 in os.listdir(folder2):
            if extract_number(filename2) == number:
                corresponding_file2 = filename2
                break

        if corresponding_file2 is None:
            print(f"No corresponding file found in {folder2} for {filename1}, skipping.")



        # Convert the text to an image
        file_path1 = os.path.join(folder1, filename1)
        image = np.loadtxt(file_path1)

        # Display the image
        
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.show(block = False)
        plt.pause(0.1)
        #plt.close()
        

        # Get user input to categorize the file
        category = input("Enter category (s -> shoe, t -> top, b -> bottom, r->bag, d->dress, g->garbage): ").strip().lower()



        # Determine the target folder based on user input
        if category == 's':
            target_folder = output_shoe
        elif category == 't':
            target_folder = output_top
        elif category == 'b':
            target_folder = output_bottom
        elif category == 'r':
            target_folder = output_bag
        elif category == 'd':
            target_folder = output_dress
        elif category == 'g':
            target_folder = output_garbage



        # Move both corresponding files from folder1 and folder2 to the target folder
        shutil.move(file_path1, os.path.join(target_folder, filename1))
        file_path2 = os.path.join(folder2, corresponding_file2)
        shutil.move(file_path2, os.path.join(target_folder, corresponding_file2))

print("Categorization complete.")