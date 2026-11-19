#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via sbatch}"
REPRO_PARENT="$1"
REPRO_AUDIT="$2"
REPRO_IMAGE="$3"
srun --container-image="$REPRO_IMAGE" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPRO_AUDIT/source" \
  python -m reproduction.synthetic.reevaluate --run "$REPRO_PARENT" --output "$REPRO_AUDIT/evaluation"
