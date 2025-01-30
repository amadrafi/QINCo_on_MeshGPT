#!/bin/bash
#SBATCH --job-name=meshgpt_training_job
#SBATCH --output=output_%j.log      # Output file (%j for job ID)
#SBATCH --account=module-mlp
#SBACTH --user=s2141524
#SBATCH --error=error_%j.log        # Error file
#SBATCH --time=2-00:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --partition=PGR-Standard                 # Partition (use appropriate partition)
#SBATCH --ntasks=1                      # Number of tasks (adjust as needed)
#SBATCH --cpus-per-task=1                 # Number of CPUs per task
#SBATCH --gpus-per-node=8

# Print job details
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_NODELIST"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "GPUs per Task: $SLURM_GPUS_PER_NODE"

# Run the Python script
#srun python meshgpt-pytorch/train_autoencoder.py
#source ../mesh/bin/activate
source ~/venv3.12/bin/activate
# srun python test_cuda.py
# srun python train_autoencoder.py
# srun python -m torch.distributed.launch --nproc_per_node=8 train_auto.py
srun python loadShapenet.py
# srun python inspect_autoencoder.py
# srun python train_transformer.py
# srun python train_2.py
# srun python inspect_transformer.py
# srun python test_mixin.py

