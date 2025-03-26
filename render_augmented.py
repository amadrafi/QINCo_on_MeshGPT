import os
import torch
import trimesh
import numpy as np
import pandas as pd
import csv
import json
from collections import OrderedDict
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer,
    MeshDataset
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
) 
import argparse  # Added for command-line arguments
from helper import get_mesh, load_shapenet, load_filename

def augment_mesh(vertices, scale_factor, rotation_range=(-np.pi/12, np.pi/12)):
    # Apply a small random jitter to each vertex
    jitter_factor = 0.01
    possible_values = np.arange(-jitter_factor, jitter_factor, 0.0005)
    offsets = np.random.choice(possible_values, size=vertices.shape)
    vertices = vertices + offsets
    
    # Apply scaling
    vertices = vertices * scale_factor
    
    # Apply random rotation about the y-axis with a smaller range (-15° to 15°)
    # angle = np.random.uniform(rotation_range[0], rotation_range[1])
    # cos_angle = np.cos(angle)
    # sin_angle = np.sin(angle)
    # rotation_matrix = np.array([
    #     [cos_angle, 0, sin_angle],
    #     [0,         1, 0],
    #     [-sin_angle, 0, cos_angle]
    # ])
    # vertices = vertices.dot(rotation_matrix.T)
    
    # Adjust the y-coordinate so the mesh remains on the "ground"
    min_y = np.min(vertices[:, 1])
    difference = -0.95 - min_y
    vertices[:, 1] += difference
    
    return vertices

def load_shapenet(directory, variations=9):
    """
    Loads the .obj file from the given directory and returns a list containing one original mesh
    and 'variations' augmented versions of that mesh (9 by default).

    Args:
        directory (str): Path to the directory containing the mesh .obj file.
        variations (int): Number of augmented versions to generate.

    Returns:
        List[Dict]: A list of dictionaries, each containing the mesh data:
            - "vertices": Tensor of vertices on GPU.
            - "faces": Tensor of faces on GPU.
            - "face_edges": Face edges derived from faces.
            - "texts": A label indicating whether the mesh is original or an augmentation.
    """
    obj_datas = []
    
    # Look for an OBJ file in the directory.
    target_file = None
    for filename in os.listdir(directory):
        if filename.lower().endswith('.obj'):
            target_file = os.path.join(directory, filename)
            break

    if target_file is None:
        print(f"[DEBUG] No OBJ file found in directory: {directory}.")
        return []
    
    # Load the original mesh using the helper function.
    vertices, faces = get_mesh(target_file)
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        print(f"[DEBUG] get_mesh failed for '{target_file}'.")
        return []
    
    # Derive face edges (assumes derive_face_edges_from_faces is available)
    face_edges = derive_face_edges_from_faces(faces)
    
    # Convert original mesh to torch tensors and add it to the list.
    original_data = {
        "vertices": torch.tensor(vertices.tolist(), dtype=torch.float).to("cuda"),
        "faces": torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"),
        "face_edges": face_edges,
        "texts": "original",
    }
    obj_datas.append(original_data)
    
    # Create 9 augmented versions.
    # We'll use a set of possible scale factors (similar to your earlier approach).
    possible_scales = np.arange(0.75, 1.0, 0.005)
    for i in range(variations):
        scale_factor = np.random.choice(possible_scales)
        # Augment the vertices; note we use a copy to avoid modifying the original.
        augmented_vertices = augment_mesh(vertices.copy(), scale_factor)
        aug_data = {
            "vertices": torch.tensor(augmented_vertices.tolist(), dtype=torch.float).to("cuda"),
            "faces": torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"),
            "face_edges": face_edges,
            "texts": f"augmented_{i+1}",
        }
        obj_datas.append(aug_data)
    
    return obj_datas

def save_variations_as_obj(obj_datas, output_path, spacing=2.0):
    """
    Combines multiple mesh variations into one OBJ file by translating each variation
    along the x-axis so that they are arranged side by side.

    Args:
        obj_datas (list): List of dictionaries each containing "vertices" and "faces".
        output_path (str): Path to the output OBJ file.
        spacing (float): Distance between each mesh along the x-axis.
    """
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for i, data in enumerate(obj_datas):
        # Convert vertices and faces from torch tensors to numpy arrays.
        vertices = data["vertices"].cpu().numpy()  # shape: (N, 3)
        faces = data["faces"].cpu().numpy()          # shape: (M, 3) typically
        
        # Translate vertices along the x-axis so that each mesh is side by side.
        translation = np.array([i * spacing, 0, 0])
        vertices_translated = vertices + translation
        
        combined_vertices.append(vertices_translated)
        
        # Adjust face indices to account for the vertices added so far.
        faces_adjusted = faces + vertex_offset
        combined_faces.append(faces_adjusted)
        
        vertex_offset += vertices.shape[0]

    # Concatenate all vertices and faces.
    combined_vertices = np.concatenate(combined_vertices, axis=0)
    combined_faces = np.concatenate(combined_faces, axis=0)
    
    # Write the combined mesh to an OBJ file.
    with open(output_path, 'w') as f:
        # Write vertices.
        for v in combined_vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # Write faces (note: OBJ format indices start at 1).
        for face in combined_faces:
            # If face is an array of indices, add 1 to each index.
            face_str = " ".join(str(idx + 1) for idx in face)
            f.write("f {}\n".format(face_str))
    
    print(f"Combined OBJ file saved to: {output_path}")

# Example usage:
# Assuming your functions get_mesh, derive_face_edges_from_faces, and augment_mesh are defined.
directory = "./shapenet/ShapeNetCore.v1/03001627/20b8c6959784f2da83b763ebf4ad2b38"
output_obj_path = "./renders/combined_variations_desk.obj"

# Load the augmented meshes.
# obj_datas = load_shapenet(directory, variations=10)
# if obj_datas:
#     # Save the 10 variations arranged side by side in one OBJ file.
#     save_variations_as_obj(obj_datas, output_obj_path, spacing=2.0)

dataset = MeshDataset.load("./shapenet/ShapeNetCore.v1/ShapeNetCore.v1_200.npz")
_, _ = dataset.faces_statistics()
