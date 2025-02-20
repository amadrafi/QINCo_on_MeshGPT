#!/bin/bash
#SBATCH --job-name=meshgpt_training_job
#SBATCH --output=logs_transformers/output_%j.log      # Output file (%j for job ID)
#SBATCH --account=module-mlp
#SBATCH --error=logs_transformers/error_%j.log        # Error file
#SBATCH --time=3-00:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --partition=Teach-Standard      # Partition (use appropriate partition)
#SBATCH --ntasks-per-node=1                      # Number of tasks (adjust as needed)
#SBATCH --nodelist=landonia25
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=4

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs per Task: $SLURM_GPUS_ON_NODE"

mkdir -p logs_transformers

export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# Run the Python script
source ~/venv3.12/bin/activate
# srun accelerate launch --multi-gpu continue_transformer_shapenet.py --quant lfq --codeSize 4096
srun accelerate launch --multi-gpu --mixed_precision=fp16 generate_mesh.py --quant lfq --codeSize 4096 --data "shapenet"


