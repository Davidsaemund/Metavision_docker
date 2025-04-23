import h5py
import matplotlib.pyplot as plt
import numpy as np

def plot_tensor_graphs_from_h5(h5_file, channel_index):
    """
    Plots graphs from an .h5 tensor file.
    
    Parameters:
        h5_file (str): Path to the .h5 tensor file.
        channel_index (int): Index of the channel (0 to 124) to select for plotting.
    """
    # Open the .h5 file
    with h5py.File(h5_file, 'r') as f:
        # Assuming the tensor is stored in a dataset named 'tensor'
        dataset_name = list(f.keys())[0]  # Automatically find the dataset name
        tensor = f[dataset_name][...]  # Load the entire tensor into memory

    # Check the shape of the tensor
    #if tensor.shape != (125, 2, 360, 640):
        #raise ValueError(f"Expected tensor shape (125, 2, 360, 640), got {tensor.shape}")
    
    # Select the specified channel
    selected_channel = tensor[channel_index]  # Shape: (2, 360, 640)

    # Generate graphs for each of the 2 channels
    for i in range(2):
        data = selected_channel[i]  # Shape: (360, 640)
        
        # Create a graph
        plt.figure(figsize=(10, 6))
        plt.imshow(data, extent=[0, 640, 360, 0], aspect='auto', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title(f"Graph for Channel {channel_index}, Sub-Channel {i}, Micro time bin 1") #
        plt.xlabel("X-axis (640)")
        plt.ylabel("Y-axis (360)")
        plt.tight_layout()
        
        # Save or display the graph
        plt.savefig(f"channel_{channel_index}_subchannel_{i}.png")
        plt.show()

# Example usage
# Provide the path to your .h5 tensor file and specify the channel index to visualize
h5_file_path = "DATASET_MOVING_EVENT_CUBE_RNN/train/moving_mid_5.h5"  # Replace with your file path
channel_to_plot = 38  # Replace with your desired channel index (0 to 124)

plot_tensor_graphs_from_h5(h5_file_path, channel_to_plot)
