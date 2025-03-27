#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot ground truth classification from a .npy file using index-based x-axis")
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Path to the .npy file containing the ground truth data.")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    # Load the .npy file using allow_pickle=True because the file stores tuples.
    data = np.load(args.ground_truth, allow_pickle=True)
    
    # Check if the loaded data is 1D (an array of tuples) or already 2D.
    if data.ndim == 1:
        # Each element is a tuple, so we extract the ground truth from index 5.
        gt_classes = np.array([row[5] for row in data]).astype(int)
    else:
        gt_classes = data[:, 5].astype(int)
    
    # Create an index-based x-axis (e.g., 0 to number of entries - 1)
    indices = np.arange(len(gt_classes))

    # Define the label map: {"0": "left", "1": "center", "2": "right"}
    label_map = {0: "left", 1: "center", 2: "right"}

    # Create the plot.
    plt.figure(figsize=(12, 6))
    # Use a step plot to visualize discrete changes in the ground truth.
    plt.step(indices, gt_classes, where='post', label="Ground Truth", marker='o')

    plt.xlabel("Frame Index")
    plt.ylabel("Class")
    plt.title("Ground Truth Classification Over Frame Index")
    # Replace numeric y-axis ticks with the corresponding label names.
    plt.yticks([0, 1, 2], [label_map[0], label_map[1], label_map[2]])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
