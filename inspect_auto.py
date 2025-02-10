import torch
import trimesh
import numpy as np
import os
import csv
import json
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
from helper import get_mesh, augment_mesh, load_shapenet, load_filename, filter_dataset

def main():
    parser = argparse.ArgumentParser(description="Mesh GPT Training Script")
    parser.add_argument("--quant", type=str, default="lfq", choices=["lfq", "qinco", "rvq"],
                        help="Type of quantization to use (default: lfq)")
    parser.add_argument("--codeSize", type=int, default=4096,
                        help="Codebook size for the mesh autoencoder (default: 4096)")
    args = parser.parse_args()

    project_name = "demo_mesh"
    working_dir = f'./{project_name}'

    quant = args.quant
    codeSize = args.codeSize

    useQinco = True if quant == "qinco" else False
    useLfq = True if quant == "lfq" else False

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Experiment: {quant} @ {codeSize} with {project_name}")
 
    working_dir = Path(working_dir)
    working_dir.mkdir(exist_ok = True, parents = True)
    dataset_path = working_dir / (project_name + ".npz")

    if not os.path.isfile(dataset_path):
        data = load_filename("./demo_mesh",50)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)

    dataset = MeshDataset.load(dataset_path)
    print(dataset.data[0].keys())

    autoencoder = MeshAutoencoder(
            decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 20 + (384,) * 6,
            codebook_size = codeSize,  # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
            dim_codebook = 192,
            dim_area_embed = 16,
            dim_coor_embed = 16,
            dim_normal_embed = 16,
            dim_angle_embed = 8,
            attn_decoder_depth  = 4,
            attn_encoder_depth = 2,
            use_qinco=useQinco,
            use_residual_lfq = useLfq,
        ).to(device)

    total_params = sum(p.numel() for p in autoencoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    print(f"Total parameters: {total_params}")
    
    dataset.data = [dict(d) for d in dataset.data] * 10
    print(f"Length of dataset: {len(dataset.data)}")

    pkg = torch.load(str(f'{working_dir}/mesh-encoder_{project_name}_{quant}_{codeSize}.ckpt.pt'), weights_only=True) 
    autoencoder.load_state_dict(pkg['model'])
    for param in autoencoder.parameters():
        param.requires_grad = True

    if accelerator.is_main_process:
        print(f"Successfully loaded model {quant} @ {codeSize}")

    min_mse, max_mse = float('inf'), float('-inf')
    min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
    random_samples, random_samples_pred, all_random_samples = [], [], []
    total_mse, sample_size = 0.0, 200

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

    print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')    
    mesh_render.combind_mesh_with_rows(f'{working_dir}\mse_rows.obj', all_random_samples)

if __name__ == "__main__":
    main()

