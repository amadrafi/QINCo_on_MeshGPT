#!/bin/bash
#SBATCH --job-name=meshgpt_training_job
#SBATCH --output=transformer_output_%j.log      # Output file (%j for job ID)
#SBATCH --account=module-mlp
#SBATCH --error=transformer_error_%j.log        # Error file
#SBATCH --time=3-00:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --partition=Teach-LongJobs            # Partition (use appropriate partition)
#SBATCH --ntasks-per-node=1                      # Number of tasks (adjust as needed)
#SBATCH --nodelist=landonia24
#SBATCH --gres=gpu:titan_x_pascal:2
#SBATCH --cpus-per-task=4

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs per Task: $SLURM_GPUS_ON_NODE"


export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

# Run the Python script
source ~/venv3.12/bin/activate
srun accelerate launch --multi-gpu --mixed_precision=fp16 train_transformer.py --quant rvq --codeSize 4096


