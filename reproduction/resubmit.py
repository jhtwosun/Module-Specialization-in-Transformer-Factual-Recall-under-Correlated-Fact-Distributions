"""Resubmit the cells of a run that produced no measurement, into a fresh run.

The cluster's backfill partition preempts, and `launch.py` passes `--no-requeue`
so a preempted cell fails visibly instead of restarting into a directory that
already holds partial results. The recovery is always the same: take the cells
that have no metrics, point at the configurations they came from, and submit them
as a new immutable run. This does that in one step, and never touches the original
run directory.

A cell counts as finished when it has written its metrics: `metrics.json` for the
synthetic and two-hop backends, any `*/metrics.json` for natural pretrained runs,
or `known_scan.json` for an availability scan.

Usage:

    python -m reproduction.resubmit --run runs/natural-llama-3.1-8b-001 \
        --output runs/natural-llama-3.1-8b-003 --minutes 360 --submit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from reproduction.launch import DEFAULT_IMAGE, ROOT


def finished(directory: Path) -> bool:
    if (directory / "metrics.json").is_file() or (directory / "known_scan.json").is_file():
        return True
    return any(directory.glob("*/metrics.json"))


def missing(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    rows = []
    for cell in manifest["cells"]:
        if not finished(run / cell["output"]):
            rows.append(cell)
    return manifest, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--partition", default="backfill")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--memory", default="64G", help="host memory per cell, matching launch.py")
    parser.add_argument("--carry-checkpoints", action="store_true",
                        help="copy the old run's checkpoints into the new run so its cells continue "
                             "from them. launch --resume reuses the original source snapshot, which is "
                             "right for immutability but means a code fix never reaches the node; this "
                             "takes a fresh snapshot and keeps the training progress")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split each natural cell's selected facts into this many disjoint cells, "
                             "so a preemption costs one chunk instead of the whole cell")
    args = parser.parse_args()

    families, configs, report = set(), [], []
    for run in args.run:
        manifest, rows = missing(run.resolve())
        report.append(f"{run}: {len(rows)} of {len(manifest['cells'])} cells without metrics")
        for cell in rows:
            origin = Path(cell["origin"])
            if not origin.is_file():
                # The origin may have been regenerated; fall back to the run's own
                # frozen copy, which is the configuration the cell actually ran.
                origin = run.resolve() / cell["config"]
            families.add(cell["family"])
            configs.append(origin)
    print("\n".join(report))
    if not configs:
        print("Nothing to resubmit")
        return
    if args.chunks > 1:
        expanded = args.output.parent / f"{args.output.name}-chunk-configs"
        expanded.mkdir(parents=True, exist_ok=True)
        split = []
        for index, config in enumerate(configs):
            body = json.loads(Path(config).read_text())
            if body.get("experiment") != "natural_counterfact":
                raise SystemExit("--chunks only applies to natural CounterFact cells")
            # Subdivide whatever partition the cell already had instead of replacing
            # it. facts[c::C] split into N parts is facts[c + C*j :: C*N], so the new
            # chunks still partition exactly the same selection and never overlap.
            have, mine = body.get("pair_chunks", 1), body.get("pair_chunk", 0)
            for part in range(args.chunks):
                chunk, chunks = mine + have * part, have * args.chunks
                target = expanded / f"{index:04d}-chunk{chunk}of{chunks}.json"
                target.write_text(json.dumps(dict(body, pair_chunks=chunks, pair_chunk=chunk),
                                             indent=2, sort_keys=True) + "\n")
                split.append(target)
        print(f"Split {len(configs)} cells into {len(split)} chunk cells in {expanded}")
        configs = split
    if len(families) != 1:
        raise SystemExit(f"Cells span several families {sorted(families)}; resubmit them separately")

    command = [sys.executable, "-m", "reproduction.launch", "--family", families.pop(),
               "--output", str(args.output), "--image", str(args.image), "--partition", args.partition,
               "--concurrency", str(args.concurrency), "--minutes", str(args.minutes),
               "--memory", args.memory]
    for config in configs:
        command += ["--config", str(config)]
    if args.submit:
        command.append("--submit")
    print(f"Resubmitting {len(configs)} cells")
    if args.carry_checkpoints:
        command_without_submit = [part for part in command if part != "--submit"]
        result = subprocess.run(command_without_submit, cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
        carried = 0
        for run in args.run:
            for cell in json.loads((run.resolve() / "manifest.json").read_text())["cells"]:
                source_cell = run.resolve() / cell["output"]
                for checkpoint in sorted(source_cell.glob("*/checkpoint.pt")):
                    target = args.output.resolve() / cell["output"] / checkpoint.parent.name
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkpoint, target / "checkpoint.pt")
                    for name in ("config.json", "provenance.json"):
                        if (source_cell / name).is_file():
                            shutil.copy2(source_cell / name, args.output.resolve() / cell["output"] / name)
                    carried += 1
        print(f"Carried {carried} checkpoints into {args.output}")
        if args.submit:
            resume = [sys.executable, "-m", "reproduction.launch", "--family", command[command.index("--family")+1],
                      "--output", str(args.output), "--resume", "--image", str(args.image),
                      "--partition", args.partition, "--concurrency", str(args.concurrency),
                      "--minutes", str(args.minutes), "--memory", args.memory, "--submit"]
            raise SystemExit(subprocess.run(resume, cwd=ROOT).returncode)
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
