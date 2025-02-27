import os
import numpy as np
import trimesh
import torch
import pandas as pd
import argparse
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# Import helper functions from helper.py
from helper import get_mesh, load_filename

def sample_points(vertices, faces, num_points=10000):
    """
    Create a trimesh object from vertices and faces and sample points on its surface.
    """
    # Create a mesh without additional processing
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def chamfer_distance_l1(points1, points2):
    """
    Compute Chamfer distance between two point clouds using the Manhattan (L1) distance.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    distances1, _ = tree2.query(points1, k=1, p=1)
    distances2, _ = tree1.query(points2, k=1, p=1)
    
    return np.mean(distances1) + np.mean(distances2)

def minimum_matching_distance(points1, points2):
    """
    Compute the optimal one-to-one matching distance (using the Hungarian algorithm)
    between two point clouds. Both point clouds must have the same number of points.
    """
    if points1.shape[0] != points2.shape[0]:
        raise ValueError("For minimum matching distance, both point clouds must have the same number of points.")
    cost_matrix = np.abs(points1[:, None, :] - points2[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

def coverage(points_ref, points_pred, threshold):
    """
    Compute the Coverage metric: fraction of points in the reference that have a neighbor
    in the predicted set within the specified Manhattan distance threshold.
    """
    tree = cKDTree(points_pred)
    distances, _ = tree.query(points_ref, k=1, p=1)
    return np.sum(distances < threshold) / len(points_ref)

def one_nn_accuracy(points_ref, points_pred):
    """
    Compute 1-Nearest Neighbor accuracy. Points from the reference and predicted point clouds
    are treated as different classes. For each point, we check whether the nearest neighbor (excluding itself)
    belongs to the same class.
    """
    X = np.concatenate([points_ref, points_pred], axis=0)
    labels = np.array([0] * len(points_ref) + [1] * len(points_pred))
    tree = cKDTree(X)
    
    correct = 0
    for i in range(len(X)):
        distances, indices = tree.query(X[i], k=2, p=1)  # first neighbor is the point itself
        nn_index = indices[1]
        if labels[i] == labels[nn_index]:
            correct += 1
    return correct / len(X)

def main():
    parser = argparse.ArgumentParser(description="Evaluate mesh quality using several metrics.")
    parser.add_argument("--mesh_dir", type=str, required=True,
                        help="Directory containing mesh files (.obj, .glb, .off)")
    parser.add_argument("--ref_mesh", type=str, default=None,
                        help="Path to a reference mesh file. If not provided, the first mesh in the directory is used.")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample from each mesh")
    parser.add_argument("--coverage_threshold", type=float, default=0.05,
                        help="Threshold for coverage metric (in the same scale as the sampled points)")
    parser.add_argument("--variations", type=int, default=1,
                        help="Number of variations to load per mesh (used by load_filename)")
    args = parser.parse_args()

    # Load the reference mesh
    if args.ref_mesh:
        vertices_ref, faces_ref = get_mesh(args.ref_mesh)
        if vertices_ref is None or faces_ref is None:
            print("Error loading reference mesh. Exiting.")
            return
        print(f"Using provided reference mesh: {args.ref_mesh}")
    else:
        # If no reference mesh is given, select the first eligible file in the directory.
        mesh_files = [f for f in os.listdir(args.mesh_dir) if f.lower().endswith((".obj", ".glb", ".off"))]
        if not mesh_files:
            print("No mesh files found in the directory.")
            return
        ref_file = os.path.join(args.mesh_dir, mesh_files[0])
        vertices_ref, faces_ref = get_mesh(ref_file)
        if vertices_ref is None or faces_ref is None:
            print(f"Error loading reference mesh from file: {ref_file}")
            return
        print(f"No reference mesh provided; using '{ref_file}' as the reference.")

    ref_points = sample_points(vertices_ref, faces_ref, args.num_points)

    # Load mesh data using helper.load_filename
    print("Loading meshes from directory...")
    mesh_data_list = load_filename(args.mesh_dir, args.variations)
    if not mesh_data_list:
        print("No meshes loaded. Exiting.")
        return

    results = []
    for i, obj_data in enumerate(mesh_data_list):
        # Convert torch tensors to NumPy arrays for sampling and evaluation
        try:
            vertices = obj_data["vertices"].cpu().numpy()
            faces = obj_data["faces"].cpu().numpy()
        except Exception as e:
            print(f"[DEBUG] Failed converting mesh '{obj_data.get('texts', 'unknown')}' to NumPy arrays: {e}")
            continue

        try:
            pred_points = sample_points(vertices, faces, args.num_points)
        except Exception as e:
            print(f"[DEBUG] Failed sampling points for mesh '{obj_data.get('texts', 'unknown')}': {e}")
            continue

        # Compute the evaluation metrics.
        try:
            chamfer = chamfer_distance_l1(ref_points, pred_points)
        except Exception as e:
            print(f"[DEBUG] Error computing Chamfer distance for '{obj_data.get('texts', 'unknown')}': {e}")
            chamfer = None

        try:
            mmd = minimum_matching_distance(ref_points, pred_points)
        except Exception as e:
            print(f"[DEBUG] Error computing Minimum Matching Distance for '{obj_data.get('texts', 'unknown')}': {e}")
            mmd = None

        try:
            cov = coverage(ref_points, pred_points, args.coverage_threshold)
        except Exception as e:
            print(f"[DEBUG] Error computing Coverage for '{obj_data.get('texts', 'unknown')}': {e}")
            cov = None

        try:
            nn_acc = one_nn_accuracy(ref_points, pred_points)
        except Exception as e:
            print(f"[DEBUG] Error computing 1-NN Accuracy for '{obj_data.get('texts', 'unknown')}': {e}")
            nn_acc = None

        results.append({
            "mesh_name": obj_data.get("texts", f"mesh_{i}"),
            "chamfer_distance": chamfer,
            "minimum_matching_distance": mmd,
            "coverage": cov,
            "one_nn_accuracy": nn_acc
        })
        print(f"[DEBUG] Processed mesh '{obj_data.get('texts', f'mesh_{i}')}'.")

    # Save the results to a CSV file.
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)
    output_csv = "mesh_evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    main()