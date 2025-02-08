import os
from pathlib import Path
from meshgpt_pytorch import MeshDataset

def extract_label_distribution(npz_path):
    """
    Loads a MeshDataset from an .npz file and prints out the
    distribution of labels (based on the 'texts' key).
    """

    # 1. Load the dataset from the NPZ file
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file not found at: {npz_path}")
    dataset = MeshDataset.load(npz_path)

    # 2. Gather all labels from the dataset
    label_counts = {}
    for i, item in enumerate(dataset.data):
        label = item.get("texts", None)  # or whatever your label field is named
        if label is None:
            # If there's no label in this entry, skip or handle differently
            continue

        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # 3. Print or return the distribution
    total_entries = sum(label_counts.values())
    print(f"Loaded dataset from: {npz_path}")
    print(f"Total labeled entries: {total_entries}\n")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label} : {count}")


def main():
    # Update this path to point to your NPZ file
    dataset_path = Path("./shapenet/ShapeNetCore.v1/ShapeNetCore.v1.npz")

    extract_label_distribution(dataset_path)

if __name__ == "__main__":
    main()
