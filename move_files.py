import os
import shutil

# Define the prefix and the destination folder
file_prefix = "simple_dense_lkrelu_"
destination_folder = "simple_dense_lkrelu_000001"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get the current working directory
current_directory = os.getcwd()

# Iterate over all files in the current directory
for filename in os.listdir(current_directory):
    if filename.startswith(file_prefix):
        # Construct full file path
        file_path = os.path.join(current_directory, filename)
        if os.path.isfile(file_path):
            # Move the file to the destination folder
            shutil.move(file_path, os.path.join(destination_folder, filename))

print(f"All files starting with '{file_prefix}' have been moved to '{destination_folder}'")
