#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot classification scores for Timesurface from a CSV file."
    )
    parser.add_argument("-p", "--prediction", type=str, required=True,
                        help="Path to the CSV file containing the classification scores.")
    args = parser.parse_args()

    # Read the CSV file specified by the user.
    df = pd.read_csv(args.prediction)

    # Create a 2x2 grid of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # Flatten the 2x2 array for easy iteration.

    # Plot each classification score on its own subplot using the 'frame' column as data,
    # but labeling the x-axis as "Time Bin".
    axs[0].plot(df['frame'], df['background'], label='Background')
    axs[0].set_title("Background")
    axs[0].set_xlabel("Time Bin")
    axs[0].set_ylabel("Score")

    axs[1].plot(df['frame'], df['left'], label='Left', color='orange')
    axs[1].set_title("Left")
    axs[1].set_xlabel("Time Bin")
    axs[1].set_ylabel("Score")

    axs[2].plot(df['frame'], df['center'], label='Center', color='green')
    axs[2].set_title("Center")
    axs[2].set_xlabel("Time Bin")
    axs[2].set_ylabel("Score")

    axs[3].plot(df['frame'], df['right'], label='Right', color='red')
    axs[3].set_title("Right")
    axs[3].set_xlabel("Time Bin")
    axs[3].set_ylabel("Score")

    # Set a common title for the figure.
    fig.suptitle("Classification Scores for Event Cube", fontsize=16)

    # Adjust layout to prevent overlap.
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
