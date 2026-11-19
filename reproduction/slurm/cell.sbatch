#!/bin/bash
# Resource arguments are supplied by reproduction.launch, not embedded here.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch, never run this payload on a login node}"
: "${SLURM_ARRAY_TASK_ID:?Expected a Slurm array}"
REPRO_RUN="$1"
REPRO_IMAGE="$2"
# $0 may be Slurm's spool copy; derive source path from the immutable run instead.
REPRO_ROOT="$REPRO_RUN/source"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$REPRO_RUN/../../cache/huggingface"
mkdir -p "$HF_HOME"
# Weights are pre-downloaded by reproduction/fetch_models.py, so a compute node never
# needs the Hub. A cache miss then fails the cell visibly instead of silently pulling
# a different revision than the one recorded in the configuration.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
srun --container-image="$REPRO_IMAGE" \
  --container-mounts="/lustre:/lustre,$REPRO_RUN:$REPRO_RUN" \
  --container-workdir="$REPRO_ROOT" \
  bash reproduction/slurm/inside.sh "$REPRO_RUN" "$SLURM_ARRAY_TASK_ID"
