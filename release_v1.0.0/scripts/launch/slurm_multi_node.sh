#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm-%j.out
#SBATCH --error=experiments/logs/slurm-%j.err

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate virtual environment
# source venv/bin/activate

# Print info
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Launch training
srun python train.py \
    --config configs/multi_node/train_distributed.yaml \
    --output_dir experiments/multi_node_run_${SLURM_JOB_ID}

echo "Training completed!"
