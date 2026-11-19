"""Login-safe immutable run preparation and Slurm submission (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT.parent / "containers/pytorch-2509.sqsh"
MODULES = {"synthetic": "reproduction.synthetic.run", "pretrained": "reproduction.pretrained.run", "twohop": "reproduction.rebuttal.twohop", "profile": "reproduction.rebuttal.counterfact"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(configs, family, output):
    """Snapshot code/configs BEFORE submission; never overwrite a run directory."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Run directory already exists: {output}")
    if family not in MODULES:
        raise ValueError(f"Unknown family: {family}")
    decoded = [json.loads(Path(p).read_text()) for p in configs]
    if not decoded or any(not isinstance(c, dict) for c in decoded):
        raise ValueError("Supply one or more JSON object configs")
    output.mkdir(parents=True)
    code = output / "source" / "reproduction"
    shutil.copytree(ROOT / "reproduction", code,
                    ignore=shutil.ignore_patterns("runs", "cache", "datasets", "plans", "references", "__pycache__"))
    (output / "configs").mkdir()
    (output / "logs").mkdir()
    rows = []
    for i, (original, cfg) in enumerate(zip(configs, decoded)):
        name = f"{i:05d}"
        relative = f"configs/{name}.json"
        (output / relative).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        rows.append({"cell": name, "family": family, "module": MODULES[family],
                     "config": relative, "config_sha256": digest(output / relative),
                     "origin": str(Path(original).resolve()), "output": f"cells/{name}"})
    files = {str(p.relative_to(output / "source")): digest(p)
             for p in (output / "source").rglob("*") if p.is_file()}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": git.stdout.strip(), "note": "Source hashes include uncommitted reconstruction code.",
                "source_sha256": files, "cells": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def submission_command(run, manifest, image, partition="backfill", concurrency=2, minutes=30, memory="64G"):
    if not 1 <= concurrency <= 8:
        raise ValueError("Concurrency must be between 1 and 8; expand only after pilots")
    if not 1 <= minutes <= 1440:
        raise ValueError("Wall time must be 1..1440 minutes; resume longer cells")
    # backfill preempts. Requeueing restarts a cell whose output directory already has
    # partial results, which the run modules refuse to reuse, so the retry fails with a
    # confusing FileExistsError instead of the real cause. --no-requeue makes a preemption
    # show up as a failed cell, and the rerun goes to a fresh run directory as designed.
    return ["sbatch", "--parsable", "--no-requeue", "--account=coreai_devtech_all", f"--partition={partition}",
            "--nodes=1", "--ntasks=1", "--gpus=1", "--cpus-per-task=8", f"--mem={memory}",
            f"--time={minutes}", "--job-name=coreai_devtech_all-memory.repro",
            f"--array=0-{len(manifest['cells'])-1}%{concurrency}",
            f"--output={run}/logs/%A_%a.out", f"--error={run}/logs/%A_%a.err",
            str(run / "source/reproduction/slurm/cell.sbatch"), str(run), str(image)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=MODULES, required=True)
    p.add_argument("--config", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--partition", default="backfill")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--memory", default="64G", help="host memory per cell; the two-hop sweep "
                                                   "caches every layer and needs more than the default")
    p.add_argument("--submit", action="store_true", help="Without this flag only prepare and print command")
    p.add_argument("--resume", action="store_true",
                   help="resubmit an existing run directory instead of preparing a new one; cells "
                        "continue from their checkpoints, which is how a long fine-tuning cell "
                        "survives a preemptible partition")
    args = p.parse_args()
    if not args.image.is_file():
        p.error(f"Image not found: {args.image}")
    run = args.output.resolve()
    if args.resume:
        # The snapshot and configurations stay exactly as the first submission wrote
        # them; only the array is submitted again.
        manifest = json.loads((run / "manifest.json").read_text())
    else:
        manifest = prepare(args.config, args.family, run)
    if not args.resume and not args.config:
        p.error("Supply --config, or --resume to continue an existing run")
    cmd = submission_command(run, manifest, args.image.resolve(), args.partition, args.concurrency,
                             args.minutes, args.memory)
    (run / "submission_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    print(shlex.join(cmd), flush=True)
    if args.submit:
        result = subprocess.run(cmd, capture_output=True, text=True)
        (run / "submission.json").write_text(json.dumps({"command": cmd, "returncode": result.returncode,
                                  "stdout": result.stdout, "stderr": result.stderr}, indent=2) + "\n")
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
