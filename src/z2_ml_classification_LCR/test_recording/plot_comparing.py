#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted classification (from CSV) and ground truth (from .npy) using frame indices. "
                    "CSV is expected to have columns: frame, background, left, center, right. "
                    "If the background value is highest for a row, that frame is ignored."
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to the CSV file containing the prediction scores.")
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Path to the .npy file containing the ground truth data.")
    args = parser.parse_args()

    # ---------------------------
    # Process the CSV predictions
    # ---------------------------
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"CSV file not found: {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    # Verify that the CSV has the expected columns
    expected_cols = ['frame', 'background', 'left', 'center', 'right']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV file must have columns: {expected_cols}")

    # Create a boolean mask: keep rows where the background value is NOT the highest
    mask = df['background'] < df[['left', 'center', 'right']].max(axis=1)
    df_filtered = df[mask].copy()

    # Compute predicted class for each remaining row by taking argmax on left, center, right
    scores = df_filtered[['left', 'center', 'right']].values
    pred_class_indices = scores.argmax(axis=1)  # 0->left, 1->center, 2->right

    # Get the frame indices from the CSV (these are used as the x-axis for the predictions)
    csv_frame_indices = df_filtered['frame'].values

    # ---------------------------
    # Process the ground truth file
    # ---------------------------
    if not os.path.exists(args.ground_truth):
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")
    gt_data = np.load(args.ground_truth, allow_pickle=True)
    
    # The file is stored as an array of tuples.
    # Extract ground truth from index 5 (0-indexed) of each tuple.
    if gt_data.ndim == 1:
        gt_classes = np.array([row[5] for row in gt_data]).astype(int)
    else:
        gt_classes = gt_data[:, 5].astype(int)
    
    # Create an index-based x-axis for ground truth (assuming ground truth has one entry per frame)
    gt_frame_indices = np.arange(len(gt_classes))

    # ---------------------------
    # Plot both on the same figure
    # ---------------------------
    label_map = {0: "left", 1: "center", 2: "right"}
    
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth first.
    # Set a high line width ("wider") and a lower zorder so that it appears underneath.
    plt.step(gt_frame_indices, gt_classes, where='post', label="Ground Truth ",
             marker='x', color='red', linewidth=8, markersize=5, zorder=1)
    
    # Plot predicted classes (filtered) from CSV on top.
    plt.step(csv_frame_indices, pred_class_indices, where='post',
             label="Predicted Event Cube", marker='o', color='blue', linestyle='', linewidth=2, markersize=4, zorder=2)

    plt.xlabel("Frame Index")
    plt.ylabel("Classification")
    plt.title("Predicted Classification vs. Ground Truth for Event Cube")
    # Set y-ticks to show class labels instead of numeric values
    plt.yticks([0, 1, 2], [label_map[0], label_map[1], label_map[2]])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
