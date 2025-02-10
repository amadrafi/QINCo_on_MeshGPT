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
    MeshDataset
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
)
import argparse  # Added for command-line arguments

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
    
    # Center the vertices
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
        
    unique_vertices = list(seen.values()) 
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
      
    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index 
                  for old_index, vertex_tuple in enumerate(vertices_as_tuples) 
                  for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) 
                  if vertex_tuple == sorted_vertex_tuple}
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces] 
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]   
    return np.array(sorted_vertices), np.array(sorted_faces)
 
def augment_mesh(vertices, scale_factor):     
    jitter_factor = 0.01 
    possible_values = np.arange(-jitter_factor, jitter_factor, 0.0005) 
    offsets = np.random.choice(possible_values, size=vertices.shape) 
    vertices = vertices + offsets   
    
    vertices = vertices * scale_factor 
    # To ensure that the mesh models are on the "ground"
    min_y = np.min(vertices[:, 1])  
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    return vertices

#load_shapenet("./shapenet", "./shapenet_csv_files", 10, 10)   
# Find the csv files with the labels in the ShapeNetCore.v1.zip, download at https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive  
def load_shapenet(directory, per_category, variations):
    obj_datas = []   
    chosen_models_count = {}    
    print(f"per_category: {per_category} variations {variations}")
    
    with open('shapenet_labels.json', 'r') as f:
        id_info = json.load(f) 
    
    possible_values = np.arange(0.75, 1.0, 0.005) 
    scale_factors = np.random.choice(possible_values, size=variations) 
    
    for category in os.listdir(directory): 
        category_path = os.path.join(directory, category)   
        if not os.path.isdir(category_path):
            continue 
        
        num_files_in_category = len(os.listdir(category_path))
        print(f"{category_path} got {num_files_in_category} files") 
        chosen_models_count[category] = 0  
        
        for filename in os.listdir(category_path):
            if filename.endswith((".obj", ".glb", ".off")):
                file_path = os.path.join(category_path, filename)
                
                if chosen_models_count[category] >= per_category:
                    break 
                if os.path.getsize(file_path) > 20 * 1024:  # 20 kb limit = less than 400-600 faces
                    continue 
                if filename[:-4] not in id_info:
                    print("Unable to find id info for ", filename)
                    continue 
                vertices, faces = get_mesh(file_path) 
                if len(faces) > 800: 
                    continue
                
                chosen_models_count[category] += 1  
                textName = id_info[filename[:-4]]   
                
                face_edges = derive_face_edges_from_faces(faces)  
                for scale_factor in scale_factors: 
                    aug_vertices = augment_mesh(vertices.copy(), scale_factor)   
                    obj_data = {
                        "vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"),
                        "faces": torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"),
                        "face_edges": face_edges,
                        "texts": textName
                    }  
                    obj_datas.append(obj_data)
                    
    print("="*25)
    print("Chosen models count for each category:")
    for category, count in chosen_models_count.items():
        print(f"{category}: {count}") 
    total_chosen_models = sum(chosen_models_count.values())
    print(f"Total number of chosen models: {total_chosen_models}")
    return obj_datas

def load_filename(directory, variations):
    obj_datas = []    
    possible_values = np.arange(0.75, 1.0, 0.005) 
    scale_factors = np.random.choice(possible_values, size=variations) 
    
    for filename in os.listdir(directory):
        if filename.endswith((".obj", ".glb", ".off")): 
            file_path = os.path.join(directory, filename) 
            vertices, faces = get_mesh(file_path)  
            
            faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
            face_edges = derive_face_edges_from_faces(faces)  
            texts, ext = os.path.splitext(filename)     
            
            for scale_factor in scale_factors: 
                aug_vertices = augment_mesh(vertices.copy(), scale_factor)  
                obj_data = {
                    "vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"),
                    "faces": faces,
                    "face_edges": face_edges,
                    "texts": texts
                } 
                obj_datas.append(obj_data)
                     
    print(f"[create_mesh_dataset] Returning {len(obj_datas)} meshes")
    return obj_datas

def main():
    # Parse command-line arguments for quant and codeSize.
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
    working_dir.mkdir(exist_ok=True, parents=True)
    dataset_path = working_dir / (project_name + ".npz")

    if not os.path.isfile(dataset_path):
        data = load_filename("./demo_mesh", 50)
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
    ckpt_path = Path(f'{working_dir}') / f'mesh-encoder_{project_name}_{quant}_{codeSize}.ckpt.pt'
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
    ) 
    
    total_params = sum(p.numel() for p in transformer.decoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    if accelerator.is_main_process:
        print(f"Decoder total parameters: {total_params}")

    labels = list(set(item["texts"] for item in dataset.data))
    dataset.embed_texts(transformer, batch_size=25)
    dataset.generate_codes(autoencoder, batch_size=50)
    if accelerator.is_main_process:
        print(dataset.data[0].keys())

    batch_size = 1  # Max 64
    grad_accum_every = 16
    
    # Set the maximal batch size (max 64) that your VRAM can handle, and use grad_accum_every to create an effective batch size of 64 (e.g., 4 * 16 = 64)
    learning_rate = 1e-2  # Start training with a higher learning rate, then lower it if stagnation occurs.
    
    trainer = MeshTransformerTrainer(
        model=transformer,
        warmup_steps=10,
        num_train_steps=100,
        dataset=dataset,
        grad_accum_every=grad_accum_every,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_every_epoch=5,
        # Uncomment below for FP16 training if desired (note: may cause nan issues)
        # accelerator_kwargs={"mixed_precision": "fp16"},
        # optimizer_kwargs={"eps": 1e-7}
    )
    loss = trainer.train(300, stop_at_loss=0.005)

    # Save the trained transformer model.
    save_path = working_dir / f"mesh-transformer_{project_name}_{quant}_{codeSize}.pt"
    trainer.save(str(save_path))   

if __name__ == "__main__":
    main()
