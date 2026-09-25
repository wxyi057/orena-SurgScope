#!/usr/bin/env bash
# Example SLURM array for data preparation, one shard per array task. Run the two stages as
# separate arrays (extraction needs every mezzanine), then write the JSONL once:
#   J=$(sbatch --parsable --array=0-7 scripts/slurm/prepare_array.sbatch transcode 8)
#   sbatch --dependency=afterok:$J --array=0-7 scripts/slurm/prepare_array.sbatch dense-mix 8
#   surgscope prepare --data heico=... --data lapchole=... --work work --regime dense-mix --jsonl-only
#SBATCH --job-name=surgscope-prepare
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
set -euo pipefail
STAGE=$1 N=$2 K=${SLURM_ARRAY_TASK_ID}
DATA=(--data heico=data/heico-focus-vqa --data lapchole=data/lapchole-focus-vqa)
if [ "$STAGE" = transcode ]; then
  surgscope transcode "${DATA[@]}" --work work --procs 16 --shard $K/$N
else
  surgscope prepare "${DATA[@]}" --work work --regime "$STAGE" --procs 32 --shard $K/$N
fi
