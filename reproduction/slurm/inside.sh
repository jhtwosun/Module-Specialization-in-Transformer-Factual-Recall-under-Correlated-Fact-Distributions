#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Batch allocation required}"
REPRO_RUN="$1"
REPRO_CELL="$2"
# Synthetic experiments only need the image's torch/numpy. Pretrained dependencies
# are installed in the ephemeral compute container, never on the login node.
REPRO_FAMILY="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][int(sys.argv[2])]["family"])' "$REPRO_RUN/manifest.json" "$REPRO_CELL")"
if [[ "$REPRO_FAMILY" == pretrained ]]; then
  python -m pip install --disable-pip-version-check -r reproduction/pretrained/requirements.txt
  python -m reproduction.pretrained.selftest
else
  python -m reproduction.synthetic.tests_gpu
fi
python -m reproduction.execute --run "$REPRO_RUN" --index "$REPRO_CELL"
