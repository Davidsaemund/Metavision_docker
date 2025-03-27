#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

def load_csv_predictions(csv_path):
    """
    Loads the CSV file and computes predicted labels.
    Expected CSV columns: frame, background, left, center, right.
    Label order:
      0: background, 1: left, 2: center, 3: right.
    """
    df = pd.read_csv(csv_path)
    expected_cols = ['frame', 'background', 'left', 'center', 'right']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV file must contain columns: {expected_cols}")
    
    # Compute predicted label by taking argmax over all 4 columns.
    probs = df[['background', 'left', 'center', 'right']].values
    pred_labels = np.argmax(probs, axis=1)
    
    return pred_labels, df['frame'].values

def load_ground_truth(np_path):
    """
    Loads the ground truth from a .npy file.
    Each element of the array is a tuple, where:
      - index 0: timestamp (not used for alignment)
      - index 5: gesture label (0: left, 1: center, 2: right)
    For each tuple, if index 5 is in {0,1,2} we shift it by +1 so that:
      left becomes 1, center becomes 2, right becomes 3.
    If not, the frame is considered background (0).
    """
    gt_data = np.load(np_path, allow_pickle=True)
    gt_labels = []
    for row in gt_data:
        raw = row[5]
        if raw in [0, 1, 2]:
            gt_labels.append(int(raw) + 1)
        else:
            gt_labels.append(0)
    return np.array(gt_labels)

def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot a confusion matrix for 4 classes: background, left, center, right.\n\n"
                    "CSV file format: columns: frame, background, left, center, right.\n"
                    "  -> Predicted labels (via argmax): 0: background, 1: left, 2: center, 3: right.\n\n"
                    "Ground truth (.npy) file: shape (1604,), each element is a tuple where index 5 is "
                    "the gesture label (0: left, 1: center, 2: right).\n"
                    "Frames with no gesture (or an invalid label) are considered background."
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to the CSV file containing prediction scores.")
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Path to the .npy file containing ground truth data.")
    args = parser.parse_args()

    # Check file existence
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"CSV file not found: {args.predictions}")
    if not os.path.exists(args.ground_truth):
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    # Load predictions and ground truth (aligned by order)
    pred_labels, frame_indices = load_csv_predictions(args.predictions)
    gt_labels = load_ground_truth(args.ground_truth)

    # Check alignment: we expect the number of predictions to match the ground truth length.
    if len(pred_labels) != len(gt_labels):
        print("Warning: Number of prediction rows and ground truth entries do not match.")
        min_len = min(len(pred_labels), len(gt_labels))
        pred_labels = pred_labels[:min_len]
        gt_labels = gt_labels[:min_len]

    # Define label space: 0: background, 1: left, 2: center, 3: right.
    labels = [0, 1, 2, 3]

    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
    
    # Compute overall accuracy
    acc = accuracy_score(gt_labels, pred_labels)
    
    # Compute per-class precision and recall (set zero_division=0 to avoid errors if no predicted samples exist for a class)
    precisions = precision_score(gt_labels, pred_labels, labels=labels, average=None, zero_division=0)
    recalls = recall_score(gt_labels, pred_labels, labels=labels, average=None, zero_division=0)

    # Define display names for each class.
    label_names = {0: "background", 1: "left", 2: "center", 3: "right"}
    class_names = [label_names[l] for l in labels]

    # Print metrics
    print(f"Overall Accuracy: {acc*100:.2f}%")
    for i, lab in enumerate(labels):
        print(f"Class {lab} ({label_names[lab]}): Precision = {precisions[i]:.2f}, Recall = {recalls[i]:.2f}")

    # Plot the confusion matrix using a Seaborn heatmap with annotations.
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
