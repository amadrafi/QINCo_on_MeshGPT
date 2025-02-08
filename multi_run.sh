#!/bin/bash
#SBATCH --job-name=meshgpt_training_job
#SBATCH --output=print_output_%j.log   # Output file (%j for job ID)
#SBATCH --error=print_error_%j.log     # Error file
#SBATCH --account=module-mlp
#SBATCH --partition=PGR-Standard
#SBATCH --time=5-00:00:00

# --------------------
# MULTI-NODE SETTINGS:
# --------------------
#SBATCH --nodes=2            # Request 2 nodes
#SBATCH --ntasks-per-node=1   # 1 task (process) per node
#SBATCH --gres=gpu:8          # 8 GPUs per node (adjust if needed)
#SBATCH --cpus-per-task=2     # Adjust CPUs per task as needed

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
echo "Total GPUs Used: $TOTAL_GPUS"

# Activate environment and configure PyTorch memory allocation
source ~/venv3.12/bin/activate
# --------------------------
# MULTI-NODE ACCELERATE RUN:
# --------------------------
# 'srun' will launch one process per node (due to --ntasks-per-node=1).
# Accelerate will detect the world size (# of nodes * gpus) automatically.
# or it can read from your accelerate config file.

# srun accelerate launch --multi_gpu --mixed_precision=fp16 train_shapenet.py 
srun accelerate launch --multi_gpu train_shapenet.py 
