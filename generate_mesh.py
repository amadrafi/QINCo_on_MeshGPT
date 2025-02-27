import torch
import trimesh
import numpy as np
import os
import csv
import json
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

##  MAIN FUNCTION
def main():
    # Parse command-line arguments for quant and codeSize.
    parser = argparse.ArgumentParser(description="Mesh GPT Training Script")
    parser.add_argument("--quant", type=str, default="lfq", choices=["lfq", "qinco", "rvq"],
                        help="Type of quantization to use (default: lfq)")
    parser.add_argument("--codeSize", type=int, default=4096,
                        help="Codebook size for the mesh autoencoder (default: 4096)")
    parser.add_argument("--data", type=str, default='demo_mesh', choices=["demo_mesh", "shapenet"],
                        help="Please choose choose the correct data set")
    args = parser.parse_args()

    quant = args.quant
    codeSize = args.codeSize

    useQinco = True if quant == "qinco" else False
    useLfq = True if quant == "lfq" else False

    accelerator = Accelerator()
    device = accelerator.device

    if args.data == "demo_mesh":
        project_name = "demo_mesh"
    elif args.data == "shapenet":
        project_name = "shapenet/ShapeNetCore.v1"
    
    if accelerator.is_main_process:
        print(f"Experiment: {quant} @ {codeSize} with {project_name}")
    
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

    if accelerator.is_main_process:
        print(f"Successfully loaded model {quant} @ {codeSize}")

    torch.cuda.empty_cache()
    gc.collect()   
    
    max_seq = max(len(d["faces"]) for d in dataset if "faces" in d) * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers) 
    if accelerator.is_main_process:
        print("Max token sequence:", max_seq)  
    
    # GPT2-Small style transformer
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
    
    total_params = sum(p.numel() for p in transformer.decoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    if accelerator.is_main_process:
        print(f"Decoder total parameters: {total_params}")

    labels = list(set(item["texts"] for item in dataset.data))
    dataset.embed_texts(transformer, batch_size=25)
    dataset.generate_codes(autoencoder, batch_size=50)
    if accelerator.is_main_process:
        print(dataset.data[0].keys())

    if args.data == "demo_mesh":
        pkg = torch.load(str(f'{working_dir}/mesh-transformer_{project_name}_{quant}_{codeSize}.pt'), weights_only=True) 
    elif args.data == "shapenet":
        pkg = torch.load(str(f'{working_dir}/mesh-transformer_shapenet_{quant}_{codeSize}.ckpt.pt'), weights_only=True) 
    transformer.load_state_dict(pkg['model'])

    if accelerator.is_main_process:
        print(f"Successfully loaded model {quant} @ {codeSize}")

    folder = './renders'
    obj_file_path = Path(folder)
    obj_file_path.mkdir(exist_ok = True, parents = True)  
    
    text_coords = [] 
    for text in labels[:10]:
        print(f"Generating {text}") 
        text_coords.append(transformer.generate(texts = [text],  temperature = 0.4))   
        
    mesh_render.save_rendering(f'{folder}/results_{args.data}_{quant}_{codeSize}.obj', text_coords)

if __name__ == "__main__":
    main()
