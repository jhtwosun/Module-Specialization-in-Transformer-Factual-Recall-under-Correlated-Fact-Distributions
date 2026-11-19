#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
REPRO_IMAGE="$1"
REPRO_PROJECT="$2"
srun --container-image="$REPRO_IMAGE" --container-mounts="/lustre:/lustre" \
  --container-workdir="$REPRO_PROJECT" \
  python -c 'import json,platform,torch; assert torch.cuda.is_available(); x=torch.arange(16,device="cuda"); assert int(x.sum())==120; print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"gpus":torch.cuda.device_count(),"status":"cuda-smoke-ok"}))'
