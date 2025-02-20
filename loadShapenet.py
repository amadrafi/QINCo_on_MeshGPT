import torch
import trimesh
import numpy as np
import pandas as pd
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
    project_name = "shapenet/ShapeNetCore.v1"

    working_dir = f'./{project_name}'

    working_dir = Path(working_dir)
    working_dir.mkdir(exist_ok = True, parents = True)
    dataset_path = working_dir / ("ShapeNetCore.v1.npz")

    if not os.path.isfile(dataset_path):
        data = load_shapenet("./shapenet/ShapeNetCore.v1", 50, 5)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)

    dataset = MeshDataset.load(dataset_path)
    print(dataset.data[0].keys())


if __name__ == "__main__":
    main()

