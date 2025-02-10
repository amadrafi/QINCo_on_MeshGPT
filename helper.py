# helper.py

import torch
import trimesh
import numpy as np
import os
import json
from collections import OrderedDict
from meshgpt_pytorch.data import derive_face_edges_from_faces

def get_mesh(file_path): 
    mesh = trimesh.load(file_path, force='mesh') 
    vertices = mesh.vertices.tolist()
    if ".off" in file_path:  # ModelNet dataset
       mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]] 
       rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
       mesh.apply_transform(rotation_matrix) 
        # Extract vertices and faces from the rotated mesh
       vertices = mesh.vertices.tolist()
            
    faces = mesh.faces.tolist()
    
    centered_vertices = vertices - np.mean(vertices, axis=0)  
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)     # Limit vertices to [-0.95, 0.95]
      
    min_y = np.min(vertices[:, 1]) 
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    
    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]   
 
    seen = OrderedDict()
    for point in vertices: 
      key = tuple(point)
      if key not in seen:
        seen[key] = point
        
    unique_vertices =  list(seen.values()) 
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
      
    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if vertex_tuple == sorted_vertex_tuple} 
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces] 
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]   
    return np.array(sorted_vertices), np.array(sorted_faces)
 
 

def augment_mesh(vertices, scale_factor):     
    jitter_factor=0.01 
    possible_values = np.arange(-jitter_factor, jitter_factor , 0.0005) 
    offsets = np.random.choice(possible_values, size=vertices.shape) 
    vertices = vertices + offsets   
    
    vertices = vertices * scale_factor 
    # To ensure that the mesh models are on the "ground"
    min_y = np.min(vertices[:, 1])  
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    return vertices

def load_shapenet(directory, per_category, variations):
    obj_datas = []
    chosen_models_count = {}
    print(f"per_category: {per_category}, variations: {variations}")

    metadata_file = os.path.join(directory, "shapenet_labels.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            id_info = json.load(f)
    else:
        print(f"[DEBUG] No JSON metadata file found at: {metadata_file}.")
        id_info = {}

    possible_values = np.arange(0.75, 1.0, 0.005)
    scale_factors = np.random.choice(possible_values, size=variations) if variations > 0 else []

    # Iterate through each category folder
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)

        # Skip non-directories at top level
        if not os.path.isdir(category_path):
            continue

        # Find corresponding .csv metadata file
        csv_file = os.path.join(directory, f"{category}.csv")
        if not os.path.exists(csv_file):
            print(f"Warning: No metadata CSV for category '{category}'")
            continue

        # Load metadata from CSV
        try:
            _ = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print(f"Warning: The file {csv_file} is empty or not a valid CSV. Skipping.")
            continue

        # Initialize count of chosen models for this category
        chosen_models_count[category] = 0

        # but you can add a debug statement if needed.
        print(f"[DEBUG] Now scanning (recursively) '{category_path}' ...")

        # Flag to stop scanning if we've hit the per_category limit
        done_category = False

        # Recursively walk through subdirectories
        for root, dirs, files in os.walk(category_path):
            if done_category:
                break  # If we've already hit the limit, no need to keep walking

            for filename in files:
                # Check file extension
                if not filename.lower().endswith((".obj", ".glb", ".off")):
                    continue

                file_path = os.path.join(root, filename)

                # Check if we reached the per-category limit
                if chosen_models_count[category] >= per_category:
                    print(f"[DEBUG] Reached per-category limit ({per_category}) for '{category}'.")
                    done_category = True
                    break

                # Check file size limit
                file_size = os.path.getsize(file_path)
                if file_size > 200  * 1024:  # 20 KB
                    print(f"[DEBUG] Skipping '{file_path}' due to size ({file_size / 1024:.2f} KB).")
                    break

                # Check if JSON metadata has an entry for this file’s base name (minus extension)
                base_name = os.path.splitext(filename)[0]
                if base_name not in id_info:
                    print(f"[DEBUG] Skipping '{file_path}' -> No ID info in JSON for '{base_name}'.")
                    continue

                # Load mesh (vertices, faces) - implement get_mesh yourself
                vertices, faces = get_mesh(file_path)
                if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
                    print(f"[DEBUG] get_mesh failed for '{file_path}' (vertices/faces are None or empty).")
                    continue

                # Check face count limit
                if len(faces) > 200:
                    print(f"[DEBUG] Skipping '{file_path}' because it has too many faces ({len(faces)}).")
                    break

                # If we reached here, the model is eligible
                chosen_models_count[category] += 1
                print(f"[DEBUG] Selected '{file_path}' for '{category}' "
                      f"(Count: {chosen_models_count[category]}).")

                # Retrieve metadata (e.g., textName from JSON)
                textName = id_info.get(base_name, "Unknown")

                # Derive face edges (implement this helper yourself)
                face_edges = derive_face_edges_from_faces(faces)

                # Create augmented data for each scale factor
                for scale_factor in scale_factors:
                    aug_vertices = augment_mesh(vertices.copy(), scale_factor)  # Implement this helper
                    obj_data = {
                        "vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"),
                        "faces": torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"),
                        "face_edges": face_edges,
                        "texts": textName,
                    }
                    obj_datas.append(obj_data)
        # End of recursive os.walk for this category

    # Print summary
    print("=" * 25)
    print("Chosen models count for each category:")
    for category, count in chosen_models_count.items():
        print(f"{category}: {count}")
    total_chosen_models = sum(chosen_models_count.values())
    print(f"Total number of chosen models: {total_chosen_models}")
    return obj_datas

def load_filename(directory, variations):
    obj_datas = []    
    possible_values = np.arange(0.75, 1.0 , 0.005) 
    scale_factors = np.random.choice(possible_values, size=variations) 
    
    for filename in os.listdir(directory):
        if filename.endswith((".obj", ".glb", ".off")): 
            file_path = os.path.join(directory, filename) 
            vertices, faces = get_mesh(file_path)  
            
            faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
            face_edges =  derive_face_edges_from_faces(faces)  
            texts, ext = os.path.splitext(filename)     
            
            for scale_factor in scale_factors: 
                aug_vertices = augment_mesh(vertices.copy(), scale_factor)  
                obj_data = {"vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"), "faces":  faces, "face_edges" : face_edges, "texts": texts } 
                obj_datas.append(obj_data)
                     
    print(f"[create_mesh_dataset] Returning {len(obj_data)} meshes")
    return obj_datas

def filter_dataset(dataset, unique_labels = False):
    unique_dicts = []
    unique_tensors = set()
    texts = set()
    for d in dataset.data:
        tensor = d["faces"]
        tensor_tuple = tuple(tensor.cpu().numpy().flatten())
        if unique_labels and d['texts'] in texts:
            continue
        if tensor_tuple not in unique_tensors:
            unique_tensors.add(tensor_tuple)
            unique_dicts.append(d)
            texts.add(d['texts'])
    return unique_dicts 
