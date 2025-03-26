import torch
import trimesh
import numpy as np
import os
import csv
import random
import json
from tqdm import tqdm
import gc
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer,
    MeshDataset,
    mesh_render
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
)
import argparse  # Added for command-line arguments
from helper import get_mesh, augment_mesh, load_shapenet, load_filename

from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

random.seed(42)
np.random.seed(42)

def chamfer_distance_l1(points1, points2):
    """
    Compute the Chamfer distance between two point clouds using the Manhattan (L1) distance.
    """
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    distances1, _ = tree2.query(points1, k=1, p=1)
    distances2, _ = tree1.query(points2, k=1, p=1)
    return np.mean(distances1) + np.mean(distances2)

def minimum_matching_distance(points1, points2):
    """
    Compute the minimum matching (optimal assignment) distance between two point clouds using the Hungarian algorithm.
    If the two point clouds have a different number of points, we sample the minimum number of points.
    """
    # Ensure both point clouds have the same number of points
    n_points = min(points1.shape[0], points2.shape[0])
    points1 = points1[:n_points]
    points2 = points2[:n_points]
    
    # Compute cost matrix using Manhattan (L1) distances.
    cost_matrix = np.abs(points1[:, None, :] - points2[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

def coverage(points_ref, points_pred, threshold):
    """
    Compute the Coverage metric:
    The fraction of points in points_ref that have at least one point in points_pred
    within the given Manhattan (L1) distance threshold.
    """
    tree = cKDTree(points_pred)
    distances, _ = tree.query(points_ref, k=1, p=1)
    return np.sum(distances < threshold) / len(points_ref)

def one_nn_accuracy(points_ref, points_pred):
    """
    Compute 1-Nearest Neighbor accuracy.
    Treats points from points_ref and points_pred as belonging to two classes.
    For each point, if its nearest neighbor (other than itself) has the same label,
    count it as correct.
    """
    X = np.concatenate([points_ref, points_pred], axis=0)
    labels = np.array([0] * len(points_ref) + [1] * len(points_pred))
    tree = cKDTree(X)
    correct = 0
    for i in range(len(X)):
        distances, indices = tree.query(X[i], k=2, p=1)  # first neighbor is itself
        nn_index = indices[1]
        if labels[i] == labels[nn_index]:
            correct += 1
    return correct / len(X)

def main():
    parser = argparse.ArgumentParser(description="Mesh GPT Training Script")
    parser.add_argument("--quant", type=str, default="lfq", choices=["lfq", "qinco", "rvq"],
                        help="Type of quantization to use (default: lfq)")
    parser.add_argument("--codeSize", type=int, default=4096,
                        help="Codebook size for the mesh autoencoder (default: 4096)")
    parser.add_argument("--data", type=str, default='demo_mesh', choices=["demo_mesh", "shapenet"],
                        help="Please choose choose the correct data set")
    parser.add_argument("--coverage_threshold", type=float, default=0.05,
                        help="Threshold for the coverage metric (default: 0.05)")
    args = parser.parse_args()

    quant = args.quant
    codeSize = args.codeSize
    whichData = args.data

    useQinco = True if quant == "qinco" else False
    useLfq = True if quant == "lfq" else False

    accelerator = Accelerator()
    device = accelerator.device

    if args.data == "demo_mesh":
        project_name = "demo_mesh"
    elif args.data == "shapenet":
        project_name = "shapenet/ShapeNetCore.v1"
    
    working_dir = f'./{project_name}'
    working_dir = Path(working_dir)
    working_dir.mkdir(exist_ok=True, parents=True)

    if args.data == "demo_mesh":
        dataset_path = working_dir / (project_name + ".npz")
    elif args.data == "shapenet":
        dataset_path = working_dir / ("ShapeNetCore.v1_200.npz")

    if not os.path.isfile(dataset_path):
        if args.data == "demo_mesh":
            data = load_filename("./demo_mesh", 50)
        elif args.data == "shapenet":
            data = load_shapenet("./shapenet/ShapeNetCore.v1", 50, 10)
        
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)

    dataset = MeshDataset.load(dataset_path)
    print(dataset.data[0].keys())

    #LOAD AUTOENCODER
    autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 20 + (384,) * 6,
        codebook_size=codeSize,  # Smaller vocab size will speed up transformer training; for larger meshes consider larger sizes.
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2,
        use_qinco=useQinco,
        use_residual_lfq=useLfq,
    ).to(device)

    total_params = sum(p.numel() for p in autoencoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    
    # Increase the dataset size by replicating it (if desired)
    dataset.data = [dict(d) for d in dataset.data] * 10
    if accelerator.is_main_process:
        print(f"Total parameters: {total_params}")
        print(f"Length of dataset: {len(dataset.data)}")

    # Load the pre-trained autoencoder weights.
    if args.data == "demo_mesh":
        ckpt_path = Path(f'{working_dir}') / f'mesh-encoder_{project_name}_{quant}_{codeSize}.ckpt.pt'
    elif args.data == "shapenet":
        ckpt_path = Path(f'{working_dir}') / f'mesh-encoder_shapenet_{quant}_{codeSize}.ckpt.pt'
    pkg = torch.load(str(ckpt_path), weights_only=True) 
    autoencoder.load_state_dict(pkg['model'])
    for param in autoencoder.parameters():
        param.requires_grad = True

    min_mse, max_mse = float('inf'), float('-inf')
    min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
    random_samples, random_samples_pred, all_random_samples = [], [], []
    total_mse, sample_size = 0.0, 200

    # To compute average face counts
    total_orig_faces = 0
    total_rec_faces = 0

    # To compute chamfer distance statistics
    total_chamfer = 0.0
    min_chamfer = float('inf')
    max_chamfer = float('-inf')

    # For Minimum Matching distance statistics
    total_min_match = 0.0
    min_min_match = float('inf')
    max_min_match = float('-inf')

    total_coverage = 0.0
    min_coverage = float('inf')
    max_coverage = float('-inf')

    total_one_nn = 0.0
    min_one_nn = float('inf')
    max_one_nn = float('-inf')

    random.shuffle(dataset.data)

    for item in tqdm(dataset.data[:sample_size]):

        codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges'])    
        codes = codes.flatten().unsqueeze(0)
        codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
    
        coords, mask = autoencoder.decode_from_codes_to_faces(codes)
        orgs = item['vertices'][item['faces']].unsqueeze(0)

        mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
        total_mse += mse 

        if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
        if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
    
        if len(random_samples) <= 30:
            random_samples.append(coords)
            random_samples_pred.append(orgs)
        else:
            all_random_samples.extend([random_samples_pred, random_samples])
            random_samples, random_samples_pred = [], []

        # Print the face counts for this item:
        orig_face_count = item['faces'].shape[0]
        rec_face_count = coords.shape[1]  # assuming coords shape is (1, num_faces, 3)
        print(f"Mesh '{item.get('texts', 'unknown')}'") #  -> Original faces: {orig_face_count}, Reconstructed faces: {rec_face_count}
        total_orig_faces += orig_face_count
        total_rec_faces += rec_face_count

        org_points = orgs.view(-1, 3).cpu().numpy()
        rec_points = coords.view(-1, 3).cpu().numpy()
        chamfer = chamfer_distance_l1(org_points, rec_points)
        total_chamfer += chamfer
        if chamfer < min_chamfer:
            min_chamfer = chamfer
        if chamfer > max_chamfer:
            max_chamfer = chamfer
        print(f"Chamfer distance for this mesh: {chamfer:.6f}")

        # Compute Minimum Matching distance between original and reconstructed points.
        min_match = minimum_matching_distance(org_points, rec_points)
        total_min_match += min_match
        if min_match < min_min_match:
            min_min_match = min_match
        if min_match > max_min_match:
            max_min_match = min_match
        print(f"Minimum matching distance for this mesh: {min_match:.6f}")

        # Coverage metric.
        cov = coverage(org_points, rec_points, args.coverage_threshold)
        total_coverage += cov
        if cov < min_coverage:
            min_coverage = cov
        if cov > max_coverage:
            max_coverage = cov
        print(f"Coverage: {cov:.6f}")

        # 1-Nearest Neighbor Accuracy.
        one_nn = one_nn_accuracy(org_points, rec_points)
        total_one_nn += one_nn
        if one_nn < min_one_nn:
            min_one_nn = one_nn
        if one_nn > max_one_nn:
            max_one_nn = one_nn
        print(f"1-NN Accuracy: {one_nn:.6f}")


    avg_orig_faces = total_orig_faces / sample_size
    avg_rec_faces = total_rec_faces / sample_size
    avg_chamfer = total_chamfer / sample_size
    avg_min_match = total_min_match / sample_size
    avg_coverage = total_coverage / sample_size
    avg_one_nn = total_one_nn / sample_size



    print(f'{20 * '#'} RESULTS {20 * '#'}')
    print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}') 
    # print(f"Average number of faces in original meshes: {avg_orig_faces:.2f}")
    # print(f"Average number of faces after reconstruction: {avg_rec_faces:.2f}")
    print(f"Chamfer distance statistics -> Average: {avg_chamfer:.6f}, Min: {min_chamfer:.6f}, Max: {max_chamfer:.6f}")
    print(f"Minimum matching distance statistics -> Average: {avg_min_match:.6f}, Min: {min_min_match:.6f}, Max: {max_min_match:.6f}")
    print(f"Coverage -> AVG: {avg_coverage:.6f}, Min: {min_coverage:.6f}, Max: {max_coverage:.6f}")
    print(f"1-NN Accuracy -> AVG: {avg_one_nn:.6f}, Min: {min_one_nn:.6f}, Max: {max_one_nn:.6f}")

    mesh_render.combind_mesh_with_rows(f'./renders/mse_rows_{whichData}_{quant}_{codeSize}_7.obj', all_random_samples)
    print(f"Saved rendering at: ./renders/mse_rows_{whichData}_{quant}_{codeSize}_final.obj")   

if __name__ == "__main__":
    main()
