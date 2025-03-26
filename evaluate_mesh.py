import os
import numpy as np
import trimesh
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import random
import gc
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# Accelerate and MeshGPT imports
from accelerate import Accelerator
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    MeshDataset,
    mesh_render
)
from meshgpt_pytorch.data import derive_face_edges_from_faces  # if needed
from helper import get_mesh, load_shapenet, load_filename

def sample_points(vertices, faces, num_points=10000):
    """
    Create a trimesh object from vertices and faces and sample points on its surface.
    """
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
    are treated as different classes.
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Generate meshes from transformer and evaluate against ground truth.")
    # Model and dataset parameters
    parser.add_argument("--quant", type=str, default="lfq", choices=["lfq", "qinco", "rvq"],
                        help="Type of quantization to use (default: lfq)")
    parser.add_argument("--codeSize", type=int, default=4096,
                        help="Codebook size for the mesh autoencoder (default: 4096)")
    parser.add_argument("--data", type=str, default='demo_mesh', choices=["demo_mesh", "shapenet"],
                        help="Choose the correct data set")
    # Evaluation parameters
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample from each mesh for evaluation")
    parser.add_argument("--coverage_threshold", type=float, default=0.05,
                        help="Threshold for coverage metric (in the same scale as the sampled points)")
    parser.add_argument("--num_examples", type=int, default=1,
                        help="Number of text-conditioned examples to evaluate")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    accelerator = Accelerator()
    device = accelerator.device

    # Define project and working directories
    if args.data == "demo_mesh":
        project_name = "demo_mesh"
    elif args.data == "shapenet":
        project_name = "shapenet/ShapeNetCore.v1"
    
    working_dir = Path(f'./{project_name}')
    working_dir.mkdir(exist_ok=True, parents=True)

    # Set dataset path and create dataset if not available.
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
    if accelerator.is_main_process:
        print(dataset.data[0].keys())
        print(f"Loaded dataset with {len(dataset.data)} meshes.")

    # Initialize autoencoder and load its checkpoint.
    quant = args.quant
    codeSize = args.codeSize
    useQinco = True if quant == "qinco" else False
    useLfq = True if quant == "lfq" else False

    autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 20 + (384,) * 6,
        codebook_size=codeSize,
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

    if args.data == "demo_mesh":
        ckpt_path = working_dir / f'mesh-encoder_{project_name}_{quant}_{codeSize}.ckpt.pt'
    elif args.data == "shapenet":
        ckpt_path = working_dir / f'mesh-encoder_shapenet_{quant}_{codeSize}.ckpt.pt'
    pkg = torch.load(str(ckpt_path), weights_only=True)
    autoencoder.load_state_dict(pkg['model'])
    for param in autoencoder.parameters():
        param.requires_grad = True
    torch.cuda.empty_cache()
    gc.collect()

    # Compute maximum sequence length required.
    max_seq = max(len(d["faces"]) for d in dataset if "faces" in d) * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers)
    
    # Initialize transformer.
    transformer = MeshTransformer(
        autoencoder,
        dim=768,
        coarse_pre_gateloop_depth=3,
        fine_pre_gateloop_depth=3,
        attn_depth=12,
        attn_heads=12,
        max_seq_len=max_seq,
        condition_on_text=True,
        gateloop_use_heinsen=False,
        dropout=0.0,
        text_condition_model_types="bge",
        text_condition_cond_drop_prob=0.0
    ).to(device)
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in transformer.decoder.parameters())
        print(f"Transformer decoder parameters: {total_params / 1e6:.1f}M")

    # Get unique text labels and shuffle.
    labels = sorted(set(item["texts"] for item in dataset.data))
    rng = random.Random(42)
    rng.shuffle(labels)
    
    # Load transformer checkpoint.
    if args.data == "demo_mesh":
        pkg = torch.load(str(working_dir / f'mesh-transformer_{project_name}_{quant}_{codeSize}.pt'), weights_only=True)
    elif args.data == "shapenet":
        pkg = torch.load(str(working_dir / f'mesh-transformer_shapenet_{quant}_{codeSize}.ckpt.pt'), weights_only=True)
    transformer.load_state_dict(pkg['model'])
    if accelerator.is_main_process:
        print(f"Successfully loaded transformer model {quant} @ {codeSize}")
    

    random.shuffle(dataset.data)

    # Loop over dataset items.
    total_mse = 0.0
    min_mse = float('inf')
    max_mse = -float('inf')
    min_coords, min_orgs = None, None
    max_coords, max_orgs = None, None

    total_orig_faces = 0
    total_rec_faces = 0
    total_chamfer = 0.0
    total_min_match = 0.0
    total_coverage = 0.0
    total_one_nn = 0.0

    min_chamfer = float('inf')
    max_chamfer = -float('inf')
    min_min_match = float('inf')
    max_min_match = -float('inf')
    min_coverage = float('inf')
    max_coverage = -float('inf')
    min_one_nn = float('inf')
    max_one_nn = -float('inf')

    random_samples = []
    random_samples_pred = []
    all_random_samples = []
    generated_meshes = []

    sample_size = 10  # Evaluate on the first 10 items
    results = []

    labels = ["sofa"]

    for label in labels:
        # Convert the text to a plain Python string (if needed)
        # text_input = str(item['texts'])
        text_input = label
        print(f"Generating mesh for text: {text_input}")
        # Generate mesh using the transformer (output is a tensor)
        try:
            result = transformer.generate(texts=[text_input], temperature=0.1)
            # If the result is a tuple, unpack it:
            generated_meshes.append(result)
            if isinstance(result, tuple):
                pred_mesh = result[0]
            else:
                pred_mesh = result
        except Exception as e:
            print(f"[DEBUG] Error generating mesh for text '{text_input}': {e}")
            continue

        # In this setup, generated_mesh is already a tensor with shape e.g. (1, num_faces, 3, 3)  # Use it directly

        # Get the original (ground truth) mesh.
        try:
            # Assume item['vertices'] is (num_vertices, 3) and item['faces'] is (num_faces, 3)
            # Index vertices with faces and add a batch dimension.
            # item = data for data in dataset.data if data['texts'] == "chair"
            item = next((data for data in dataset.data if data['texts'] == "chair"), None)
            orgs = item['vertices'][item['faces']].unsqueeze(0)
        except Exception as e:
            print(f"[DEBUG] Error processing original mesh for '{text_input}': {e}")
            continue

        # Compute Mean Squared Error between original and predicted meshes.

        # Collect a few random samples for rendering.
        if len(random_samples) <= 30:
            random_samples.append(pred_mesh)
            random_samples_pred.append(orgs)
        else:
            all_random_samples.extend([random_samples_pred, random_samples])
            random_samples, random_samples_pred = [], []

        # Print face counts.
        orig_face_count = item['faces'].shape[0]
        # Assuming pred_mesh has shape (1, num_faces, 3, 3)
        rec_face_count = pred_mesh.shape[1]
        print(f"Mesh '{text_input}' -> Original faces: {orig_face_count}, Reconstructed faces: {rec_face_count}")
        total_orig_faces += orig_face_count
        total_rec_faces += rec_face_count

        # Convert meshes to point clouds.
        org_points = orgs.view(-1, 3).cpu().numpy()
        rec_points = pred_mesh.view(-1, 3).cpu().numpy()

        # Compute Chamfer distance.
        try:
            chamfer = chamfer_distance_l1(org_points, rec_points)
        except Exception as e:
            print(f"[DEBUG] Chamfer computation error for '{text_input}': {e}")
            chamfer = 0.0
        total_chamfer += chamfer
        min_chamfer = min(min_chamfer, chamfer)
        max_chamfer = max(max_chamfer, chamfer)
        print(f"Chamfer distance: {chamfer:.6f}")

        # Compute Minimum Matching Distance.
        try:
            min_match = minimum_matching_distance(org_points, rec_points)
        except Exception as e:
            print(f"[DEBUG] Minimum matching error for '{text_input}': {e}")
            min_match = 0.0
        total_min_match += min_match
        min_min_match = min(min_min_match, min_match)
        max_min_match = max(max_min_match, min_match)
        print(f"Minimum matching distance: {min_match:.6f}")

        # Compute Coverage.
        try:
            cov = coverage(org_points, rec_points, args.coverage_threshold)
        except Exception as e:
            print(f"[DEBUG] Coverage computation error for '{text_input}': {e}")
            cov = 0.0
        total_coverage += cov
        min_coverage = min(min_coverage, cov)
        max_coverage = max(max_coverage, cov)
        print(f"Coverage: {cov:.6f}")

        # Compute 1-NN Accuracy.
        try:
            one_nn = one_nn_accuracy(org_points, rec_points)
        except Exception as e:
            print(f"[DEBUG] 1-NN computation error for '{text_input}': {e}")
            one_nn = 0.0
        total_one_nn += one_nn
        min_one_nn = min(min_one_nn, one_nn)
        max_one_nn = max(max_one_nn, one_nn)
        print(f"1-NN Accuracy: {one_nn:.6f}")

        results.append({
            "Text": text_input,
            "ChamferDistance": chamfer,
            "MinMatchingDistance": min_match,
            "Coverage": cov,
            "OneNN": one_nn,
            "OriginalFaces": orig_face_count,
            "ReconstructedFaces": rec_face_count
        })

    # Compute average metrics.
    avg_orig_faces = total_orig_faces / sample_size
    avg_rec_faces = total_rec_faces / sample_size
    avg_chamfer = total_chamfer / sample_size
    avg_min_match = total_min_match / sample_size
    avg_coverage = total_coverage / sample_size
    avg_one_nn = total_one_nn / sample_size

    print(f'\n{"#"*20} RESULTS {"#"*20}')
    # print(f'MSE -> AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
    print(f"Chamfer distance -> AVG: {avg_chamfer:.6f}, Min: {min_chamfer:.6f}, Max: {max_chamfer:.6f}")
    print(f"Minimum matching distance -> AVG: {avg_min_match:.6f}, Min: {min_min_match:.6f}, Max: {max_min_match:.6f}")
    print(f"Coverage -> AVG: {avg_coverage:.6f}, Min: {min_coverage:.6f}, Max: {max_coverage:.6f}")
    print(f"1-NN Accuracy -> AVG: {avg_one_nn:.6f}, Min: {min_one_nn:.6f}, Max: {max_one_nn:.6f}")

    csv_path = f'./metrics/evaluation_sofa_{args.data}_{args.quant}.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to: {csv_path}")

    # Combine and save a rendering of random samples.
    out_path = f'./renders/sofa_{args.data}_{args.quant}.obj'
    mesh_render.save_rendering(out_path, generated_meshes)
    print(f"Saved rendering at: {out_path}")

if __name__ == "__main__":
    main()