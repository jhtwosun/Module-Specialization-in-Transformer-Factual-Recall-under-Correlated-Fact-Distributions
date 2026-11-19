# Module Specialization in Transformer Factual Recall under Correlated Fact Distributions

## Setup

Run all commands from the repository root. Use Python 3.10+ and a CUDA GPU.
Training and evaluation require a Slurm GPU allocation.

```bash
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m unittest discover -s reproduction/tests -v
```

For pretrained-model runs, also install:

```bash
pip install -r reproduction/pretrained/requirements.txt
python -m reproduction.fetch_models
```

The model downloader stores weights in `reproduction/cache/huggingface/hub/`.
Authenticate with Hugging Face and obtain access before downloading gated models.
For natural-language experiments, download the CounterFact JSON dataset and supply
its absolute path when generating configurations.

## Configure the cluster

Before submitting jobs, adapt these settings to your cluster:

- `reproduction/launch.py`: Slurm account in `submission_command`.
- `--partition`: queue name (default: `backfill`).
- `--image`: absolute path to your PyTorch/CUDA container image.
- `reproduction/slurm/cell.sbatch`: container mounts (default: `/lustre:/lustre`).

The batch scripts use `srun --container-image`, `--container-mounts`, and
`--container-workdir`; configure the corresponding Slurm container support.
The image must contain Python, CUDA-enabled PyTorch, and NumPy. Pretrained
jobs install the additional requirements inside the container.

Set the image path for the commands below:

```bash
export REPRO_IMAGE=/absolute/path/to/pytorch.sqsh
```

## Run the starter configurations

Validate a configuration without using a GPU:

```bash
python -m reproduction.synthetic.run \
  --config reproduction/configs/pilot/subject.json --validate
```

Submit three configurations:

```bash
python -m reproduction.launch --family synthetic \
  --config reproduction/configs/pilot/independent.json \
  --config reproduction/configs/pilot/subject.json \
  --config reproduction/configs/pilot/answer.json \
  --image "$REPRO_IMAGE" --partition backfill \
  --output reproduction/runs/starter-001 \
  --concurrency 2 --minutes 30 --submit
```

Use a new output directory for each run. Omit `--submit` to prepare the run and
print the submission command. The launcher copies the code and configurations
into the run directory before submitting one GPU task per configuration.

Monitor the job ID returned by Slurm:

```bash
squeue -j JOB_ID
sacct -j JOB_ID -o JobID,State,Elapsed,ExitCode
```

Check `reproduction/runs/starter-001/logs/` and each cell's execution record
when diagnosing a failed job.

## Generate and run a configuration matrix

Synthetic families: `hierarchy`, `capacity`, `depth`, `causal`, `path`,
`threshold`, `combined`, `frozen`, `scale`.

```bash
python -m reproduction.matrix hierarchy \
  --output reproduction/configs/custom-hierarchy

configs=()
for config in reproduction/configs/custom-hierarchy/*.json; do
  configs+=(--config "$config")
done

python -m reproduction.launch --family synthetic "${configs[@]}" \
  --image "$REPRO_IMAGE" --partition backfill \
  --output reproduction/runs/hierarchy-001 \
  --concurrency 4 --minutes 30 --submit
```

The array syntax above works in Bash and Zsh. Repeat `--config` for each file;
do not pass multiple filenames to a single `--config` option.
Configuration generation requires a new directory and does not submit jobs.
Edit the generated JSON files to change seeds, training steps, or other settings.

Additional synthetic families are available through `reproduction.rebuttal.matrix`:
`decay`, `init`, `correlation`, `relation`, `joint`, `answer_convergence`.

```bash
python -m reproduction.rebuttal.matrix decay --seeds 0,1,2 \
  --output reproduction/configs/custom-decay
```

Submit those configurations with `--family synthetic` using the loop above.
For `reproduction.rebuttal.matrix twohop`, use `--family twohop`.

## Run pretrained models

Generate natural-language configurations with the CounterFact dataset:

```bash
python -m reproduction.matrix natural \
  --dataset-path /absolute/path/to/counterfact.json \
  --output reproduction/configs/custom-natural

configs=()
for config in reproduction/configs/custom-natural/*.json; do
  configs+=(--config "$config")
done

python -m reproduction.launch --family pretrained "${configs[@]}" \
  --image "$REPRO_IMAGE" --partition backfill \
  --output reproduction/runs/natural-001 \
  --concurrency 2 --minutes 300 --submit
```

Use `reproduction.matrix pythia` for synthetic fine-tuning or
`reproduction.matrix twohop` for pretrained two-hop runs; submit both with
`--family pretrained`. The separate `reproduction.rebuttal.matrix natural`
command also accepts `--dataset-path` and `--seeds`.

Verify that the model cache and dataset are accessible through the container
mounts. The batch script sets `HF_HOME` to `RUN/../../cache/huggingface` and
enables offline model loading; keep runs under `reproduction/runs/` or adapt
that path. Download the weights before submitting.

## Run within an existing GPU allocation

```bash
python -m reproduction.synthetic.tests_gpu
python -m reproduction.synthetic.run \
  --config reproduction/synthetic/configs/smoke.json \
  --output reproduction/runs/smoke-001
```

For pretrained models:

```bash
python -m reproduction.pretrained.selftest
python -m reproduction.pretrained.run \
  --config reproduction/configs/custom-natural/0000.json \
  --output reproduction/runs/natural-direct-001
```

## Resume or retry a run

Resume a synthetic checkpoint inside a GPU allocation:

```bash
python -m reproduction.synthetic.run \
  --config reproduction/configs/pilot/subject.json \
  --output reproduction/runs/subject-direct-001 --resume
```

Retry unfinished batch cells in a new run directory:

```bash
python -m reproduction.resubmit \
  --run reproduction/runs/starter-001 \
  --output reproduction/runs/starter-retry-001 \
  --image "$REPRO_IMAGE" --partition backfill \
  --concurrency 2 --minutes 30 --submit
```

Use `--carry-checkpoints` when retrying synthetic cells from their saved
checkpoints. Pretrained runs do not implement automatic checkpoint resume.

## Collect run outputs

```bash
python -m reproduction.report inspect \
  --run reproduction/runs/starter-001 --seeds 0
python -m reproduction.report aggregate \
  --run reproduction/runs/starter-001 --seeds 0 \
  --output reproduction/results/starter-001
```

Set `--seeds` to the seeds used in your run, for example `--seeds 0,1,2`.
These commands can run without a GPU. Run plotting inside a GPU allocation:

```bash
python -m reproduction.report plot \
  --run reproduction/runs/starter-001 --seeds 0 \
  --output reproduction/results/starter-plots
```

## Citation

```bibtex
@inproceedings{module_specialization_2026,
  author    = {Lee, Dong-Jae and Kim, Junmo},
  title     = {{Module Specialization in Transformer Factual Recall under Correlated Fact Distributions}},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

[MIT](LICENSE).
