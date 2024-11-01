import os
import shutil
import numpy as np

# Directory paths
source_dir = 'data'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# Ensure the directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get list of all .h5 files
all_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
all_files.sort()  # Ensure consistent order

# Shuffle the files
np.random.shuffle(all_files)

# Calculate split indices
total_files = len(all_files)
train_split = int(0.7 * total_files)
val_split = int(0.15 * total_files) + train_split

# Split the files into train, val, and test sets
train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

def move_files(file_list, destination_dir):
    for file_name in file_list:
        base_name = os.path.splitext(file_name)[0]
        h5_file = f"{base_name}.h5"
        npy_file = f"{base_name}_bbox.npy"

        # Move .h5 file
        shutil.move(os.path.join(source_dir, h5_file), os.path.join(destination_dir, h5_file))
        # Move corresponding .npy file
        shutil.move(os.path.join(source_dir, npy_file), os.path.join(destination_dir, npy_file))

# Move the files to their respective directories
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Dataset has been successfully split into train, val, and test sets.")
