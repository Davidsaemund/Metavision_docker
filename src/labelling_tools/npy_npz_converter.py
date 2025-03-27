import os
import numpy as np

def convert_npy_to_npz_with_rename(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith("_bbox.npy"):
            # Load the data from .npy file
            data = np.load(os.path.join(folder_path, filename))
            
            # Change the filename from *_bbox.npy to *_labels.npz
            new_filename = filename.replace("_bbox.npy", "_labels.npz")
            
            # Save the data as an .npz file with the new name
            np.savez(os.path.join(folder_path, new_filename), data=data)
            print(f"Converted {filename} to {new_filename}")

# Replace 'your_folder_path_here' with the path to your folder containing the .npy files
folder_path = 'val'
convert_npy_to_npz_with_rename(folder_path)
