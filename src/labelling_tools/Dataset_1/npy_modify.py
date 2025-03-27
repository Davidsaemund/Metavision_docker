import os
import numpy as np

# Define the path to the folder containing .npy files
folder_path = 'data/'

# Define the new values for the columns
new_value_2nd_col = 1  # Example new value for the 2nd column
new_value_3rd_col = 1  # Example new value for the 3rd column
new_value_4th_col = 1280  # Example new value for the 4th column
new_value_5th_col = 720  # Example new value for the 5th column

# Iterate through all .npy files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)
        
        # Replace the 2nd, 3rd, 4th, and 5th columns with the new values
        modified_data = []
        for item in data:
            modified_item = (
                item[0],  # Timestamp (unchanged)
                new_value_2nd_col,  # New value for the 2nd column
                new_value_3rd_col,  # New value for the 3rd column
                new_value_4th_col,  # New value for the 4th column
                new_value_5th_col,  # New value for the 5th column
                item[5],  # 6th column (unchanged)
                item[6],  # 7th column (unchanged)
                item[7]   # 8th column (unchanged)
            )
            modified_data.append(modified_item)
        
        # Convert modified data back to numpy array
        modified_data = np.array(modified_data, dtype=data.dtype)
        
        # Save the modified data back to the file
        np.save(file_path, modified_data)
        
        print(f"Modified file: {file_path}")
