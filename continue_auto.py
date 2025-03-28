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
    MeshDataset
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
) 
from helper import get_mesh, augment_mesh, load_shapenet, load_filename

def main():
    project_name = "demo_mesh"
    working_dir = f'./{project_name}'

    quant = "rvq"
    codeSize = 4096

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

    batch_size = 4 # The batch size should be max 64.
    grad_accum_every = 4
    # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
    learning_rate = 1e-4 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

    # Initialize Accelerator for distributed training

    autoencoder.commit_loss_weight = 0.1 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
    autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                                batch_size=batch_size,
                                                grad_accum_every = grad_accum_every,
                                                learning_rate = learning_rate,
                                                checkpoint_every_epoch=20,
                                                accelerator_kwargs={"kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=True)]},
                                            )
    loss = autoencoder_trainer.train(10000,stop_at_loss = 0.2, diplay_graph= True)

    autoencoder_trainer.save(f'{working_dir}/mesh-encoder_{project_name}_{quant}_{codeSize}.pt')

if __name__ == "__main__":
    main()

