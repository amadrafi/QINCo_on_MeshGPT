import numpy as np
import os

# Path to the NPZ file
npz_path = "./shapenet/ShapeNetCore.v1/ShapeNetCore.v1.npz"

if not os.path.isfile(npz_path):
    raise FileNotFoundError(f"NPZ file not found at: {npz_path}")

# Load the NPZ file.
# allow_pickle=True is required if you saved Python objects (like dictionaries)
data_archive = np.load(npz_path, allow_pickle=True)

# List the keys available in the NPZ file
print("Keys in the NPZ file:", data_archive.files)

# Suppose the dataset was saved under the key 'data'
if 'data' in data_archive.files:
    dataset_data = data_archive['data']  # This is likely an array (or list) of dictionaries.
    
    # Extract labels (assuming the key for labels is 'texts')
    labels = [entry['texts'] for entry in dataset_data if 'texts' in entry]
    print("Extracted labels:")
    for label in labels:
        print(label)
else:
    print("The NPZ file does not contain a 'data' key; please check its structure.")

