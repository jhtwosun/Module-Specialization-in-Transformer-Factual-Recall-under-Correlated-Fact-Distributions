"""Verify snapshot and execute one manifest cell, only inside Slurm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Heavy execution requires a Slurm allocation")
    run = a.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    row = manifest["cells"][a.index]
    for rel, sha in manifest["source_sha256"].items():
        actual = hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest()
        if actual != sha:
            raise RuntimeError(f"Snapshot modified: {rel}")
    config = run / row["config"]
    if hashlib.sha256(config.read_bytes()).hexdigest() != row["config_sha256"]:
        raise RuntimeError("Config differs from immutable manifest")
    out = run / row["output"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", row["module"], "--config", str(config), "--output", str(out)]
    if row['family'] == 'synthetic' and (out/'checkpoint.pt').exists():
        cmd.append('--resume')
    start = time.monotonic()
    status = {"cell": row["cell"], "slurm_job": os.environ["SLURM_JOB_ID"],
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": cmd}
    status_path = out / f"execution-{os.environ['SLURM_JOB_ID']}.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    proc = subprocess.run(cmd, cwd=run / "source")
    status.update(returncode=proc.returncode, elapsed_seconds=time.monotonic()-start,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"REPRO_CELL {'OK' if proc.returncode == 0 else 'FAIL'} {row['cell']}", flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
