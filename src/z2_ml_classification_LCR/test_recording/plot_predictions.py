#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot classification predictions from a CSV file after filtering out rows "
                    "where the background value is highest. The CSV file must have columns: "
                    "frame, background, left, center, right.")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to the CSV file containing the prediction scores.")
    args = parser.parse_args()

    # Check if the CSV file exists
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"CSV file not found: {args.predictions}")
    
    # Load CSV
    df = pd.read_csv(args.predictions)
    
    # Ensure the CSV contains the required columns
    expected_cols = ['frame', 'background', 'left', 'center', 'right']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV file must contain columns: {expected_cols}")

    # Filter out rows where the background column is the highest value
    # among the background, left, center, and right columns.
    # That is, we only keep rows where the background value is less than
    # the maximum of the left, center, and right values.
    mask = df['background'] < df[['left', 'center', 'right']].max(axis=1)
    df_filtered = df[mask].copy()

    # Compute the predicted class for each remaining row by taking the argmax
    # of the left, center, and right columns.
    scores = df_filtered[['left', 'center', 'right']].values
    pred_class_indices = scores.argmax(axis=1)  # 0 -> left, 1 -> center, 2 -> right

    # Use the frame column as the x-axis
    frame_indices = df_filtered['frame'].values

    # Label map for y-axis ticks
    label_map = {0: "left", 1: "center", 2: "right"}

    # Plot the predicted classification as a step plot.
    plt.figure(figsize=(12, 6))
    plt.step(frame_indices, pred_class_indices, where='post',
             marker='o', linestyle='',linewidth=1, markersize=5, label="Predicted Class")
    plt.xlabel("Frame Index")
    plt.ylabel("Predicted Class")
    plt.title("Predicted Classification for Event Cube")
    plt.yticks([0, 1, 2], [label_map[0], label_map[1], label_map[2]])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
