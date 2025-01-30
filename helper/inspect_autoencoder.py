import torch
import random
from tqdm import tqdm 
from meshgpt_pytorch import mesh_render, MeshDataset, MeshAutoencoder
from pathlib import Path
import gc
import os

project_name = "demo_mesh"
working_dir = f'./{project_name}'
type_of_vq = ""

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

#LOAD AUTOENCODER
autoencoder = MeshAutoencoder(
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        codebook_size = 1024,  # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
        dim_codebook = 120,
        dim_area_embed = 16,
        dim_coor_embed = 16,
        dim_normal_embed = 16,
        dim_angle_embed = 8,

    attn_decoder_depth  = 4,
    attn_encoder_depth = 2
).to("cuda")

pkg = torch.load(str(f'{working_dir}/mesh-encoder_{type_of_vq}.pt'))
autoencoder.load_state_dict(pkg['model'])
for param in autoencoder.parameters():
    param.requires_grad = True

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
mesh_render.combind_mesh_with_rows(f'{working_dir}/mse_rows.obj', all_random_samples)
