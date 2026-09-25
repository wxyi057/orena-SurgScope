#!/usr/bin/env bash
# Example SLURM launcher: one fine-tune on N nodes x 4 GPUs (adapt partition, GPUs per node,
# and the environment activation to your cluster).
#   sbatch -N 8 scripts/slurm/train_multinode.sbatch configs/train/dense-mix.yaml work/sft/dense-mix.jsonl runs/dense-mix
#SBATCH --job-name=surgscope-train
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
set -euo pipefail
CONFIG=$1 DATA=$2 OUT=$3
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
srun --input=none bash -c "NODE_RANK=\$SLURM_NODEID surgscope train --config $CONFIG --dataset $DATA \
    --output-dir $OUT --nproc-per-node 4 --nnodes $SLURM_NNODES"
