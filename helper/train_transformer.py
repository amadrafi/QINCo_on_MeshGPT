import gc  
import _codecs
import random
from tqdm import tqdm
import torch
import trimesh
import numpy as np
import os
import csv
import json
from collections import OrderedDict
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

# LOADING DATASET
project_name = "demo_mesh" 
working_dir = f'./{project_name}'
working_dir = Path(working_dir)
working_dir.mkdir(exist_ok = True, parents = True)
dataset_path = working_dir / (project_name + ".npz")
 
if not os.path.isfile(dataset_path):
    data = load_filename("./demo_mesh",50)  
    dataset = MeshDataset(data) 
    dataset.generate_face_edges()  
    dataset.save(dataset_path)
 
dataset = MeshDataset.load(dataset_path) 

# LOAD AUTOENCODER
autoencoder = MeshAutoencoder(
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size = 2048,  # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
        dim_codebook = 120,
        dim_area_embed = 16,
        dim_coor_embed = 16,
        dim_normal_embed = 16,
        dim_angle_embed = 8,

    attn_decoder_depth  = 4,
    attn_encoder_depth = 2
).to("cpu")
torch.serialization.add_safe_globals([_codecs.encode])
pkg = torch.load(str(f'{working_dir}' + '/mesh-encoder_'+ f'{project_name}.pt'), weights_only=True) 
autoencoder.load_state_dict(pkg['model'], strict=False)
for param in autoencoder.parameters():
    param.requires_grad = True

torch.cuda.empty_cache()
gc.collect()   
max_seq = max(len(d["faces"]) for d in dataset if "faces" in d)  * (autoencoder.num_vertices_per_face * autoencoder.num_quantizers) 
print("Max token sequence:" , max_seq)  

# GPT2-Small model INIT 
transformer = MeshTransformer(
    autoencoder,
    dim = 768,
    coarse_pre_gateloop_depth = 3,  
    fine_pre_gateloop_depth= 3,  
    attn_depth = 12,  
    attn_heads = 12,  
    max_seq_len = max_seq, 
    condition_on_text = True, 
    gateloop_use_heinsen = False,
    dropout  = 0.0,
    text_condition_model_types = "bge", 
    text_condition_cond_drop_prob = 0.0
) 

total_params = sum(p.numel() for p in transformer.decoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Decoder total parameters: {total_params}")


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
#dataset.data = filter_dataset(dataset.data, unique_labels = False)

labels = list(set(item["texts"] for item in dataset.data))
dataset.embed_texts(transformer, batch_size = 25)
dataset.generate_codes(autoencoder, batch_size = 50)
print(dataset.data[0].keys())

batch_size = 2# Max 64
grad_accum_every = 16

# Set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  4 * 16 = 64
learning_rate = 1e-3 # Start training with the learning rate at 1e-2 then lower it to 1e-3 at stagnation or at 0.5 loss.

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,num_train_steps=100, dataset = dataset,
                                 grad_accum_every=grad_accum_every,
                                 learning_rate = learning_rate,
                                 batch_size=batch_size,
                                 checkpoint_every_epoch = 15,
                                 # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                                 # However it might cause nan after a while.
                                 # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7}
                                 )
loss = trainer.train(500, stop_at_loss = 0.005)

trainer.save(f'{working_dir}\mesh-transformer_{project_name}.pt')
