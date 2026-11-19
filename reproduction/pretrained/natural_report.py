"""Aggregate natural CounterFact path-patching runs across models and relations.

``reproduction/report.py`` aggregates one run directory at a time. The R07 matrix
is submitted as one array per model, so this collects several run directories and
reports, per (model, relation), the seed mean and sample standard deviation of the
maximum edge/module ratio, together with the two numbers that decide how much that
maximum is worth: how many facts the model actually knows, and how many pairs the
cell measured.

The maximum is taken over a grid of candidate source/receiver edges inside each
seed, so a cell with few known facts maximises over noise more easily than a cell
with many. The report therefore never prints a ratio without its known-fact and
pair counts, and refuses to average cells with different pair budgets into a single
bucket number. A ``--known-scan`` directory adds the per-model relation availability
measured by ``known_scan_only`` runs.

Usage (login node, stdlib only):

    python -m reproduction.pretrained.natural_report \
        --run reproduction/runs/natural-pythia-1.4b-001 \
        --run reproduction/runs/natural-llama-3.1-8b-001 \
        --known-scan reproduction/runs/known-scan-001 \
        --output reproduction/results/natural-r07-001
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Bucket labels are the rebuttal author's assignment, not an inferred property.
# P1412 is the disclosed substitute for the absent P102 (see rebuttal/matrix.py).
AUTHOR_BUCKETS = {"P103": "subject", "P1412": "subject", "P495": "subject",
                  "P30": "answer", "P17": "answer", "P131": "answer",
                  "P19": "independent", "P27": "independent", "P407": "independent"}
REBUTTAL_CLAIM = ("answer-collision bucket highest edge/module ratio on every model, "
                  "range 5.8-26.3%, with subject and independent at 0.4-4.7%")


def read(path: Path):
    return json.loads(Path(path).read_text())


def mean_sd(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def merge(records, min_module_effect_pp=1):
    """Combine chunks of one (model, relation, seed) into a single layer grid.

    A chunk measures the same edges on a disjoint subset of the pairs, so effects
    combine as a pair-count weighted mean. The maximum is then taken once over the
    merged grid: taking it per chunk and averaging would maximise over noise in
    each chunk separately and bias the ratio upward.
    """
    grid = {}
    for record in records:
        for layer in record["layers"]:
            key = (layer["source"], layer["receiver"])
            entry = grid.setdefault(key, {"source": key[0], "receiver": key[1], "n_pairs": 0,
                                          "module": 0.0, "edge": 0.0})
            entry["n_pairs"] += layer["n_pairs"]
            entry["module"] += layer["module_effect_pp"] * layer["n_pairs"]
            entry["edge"] += layer["edge_effect_pp"] * layer["n_pairs"]
    rows = []
    for entry in grid.values():
        weight = entry["n_pairs"]
        module, edge = entry["module"] / weight, entry["edge"] / weight
        eligible = module >= min_module_effect_pp
        rows.append({"source": entry["source"], "receiver": entry["receiver"], "n_pairs": weight,
                     "module_effect_pp": module, "edge_effect_pp": edge, "eligible": eligible,
                     "edge_module_ratio": edge / module if eligible else None})
    valid = [row for row in rows if row["eligible"]]
    return rows, max(valid, key=lambda row: row["edge_module_ratio"]) if valid else None


SOURCES = {}


def collect(runs):
    chunks = {}
    for run in runs:
        run = Path(run)
        for cell in read(run / "manifest.json")["cells"]:
            directory = run / cell["output"]
            config = directory / "config.json"
            if not config.is_file():
                continue
            model = read(config)["model"]
            for metrics in sorted(directory.glob("*/metrics.json")):
                record = read(metrics)
                key = (model, record["relation"], record["seed"])
                chunks.setdefault(key, {"records": [], "runs": set()})
                chunks[key]["records"].append(record)
                chunks[key]["runs"].add(str(run))
    rows = []
    for (model, relation, seed), entry in chunks.items():
        records = entry["records"]
        grid, _ = merge(records)
        per_source = {}
        for row in grid:
            if row["eligible"] and row["edge_module_ratio"] is not None:
                ratio = 100 * row["edge_module_ratio"]
                if row["source"] not in per_source or ratio > per_source[row["source"]][0]:
                    per_source[row["source"]] = (ratio, row["receiver"])
        SOURCES[(model, relation, seed)] = per_source
        # The same cell can exist under several partitions: whole, then split into
        # four after a preemption, then into eight. Every partition covers the same
        # selected facts, so mixing them would count those pairs more than once.
        # Keep the single partition with the most of its chunks present, breaking
        # ties by pair count.
        partitions = {}
        for record in records:
            partitions.setdefault(record.get("pair_chunks") or 1, []).append(record)
        def coverage(item):
            size, group = item
            present = len({record.get("pair_chunk") or 0 for record in group})
            return (present / size, sum(record.get("pairs_emitted") or 0 for record in group))
        records = max(partitions.items(), key=coverage)[1]
        _, maximum = merge(records)
        rows.append({"model": model, "relation": relation,
                     "bucket": AUTHOR_BUCKETS.get(relation, ""), "seed": seed,
                     "known_facts": records[0]["known_facts"],
                     "pairs": sum(r.get("pairs_emitted") or 0 for r in records),
                     "chunks": len(records),
                     "max_edge_module_pct": 100 * maximum["edge_module_ratio"] if maximum else None,
                     "source": maximum["source"] if maximum else None,
                     "receiver": maximum["receiver"] if maximum else None,
                     "module_effect_pp": maximum["module_effect_pp"] if maximum else None,
                     "run": ";".join(sorted(entry["runs"]))})
    return rows


def grouped(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["relation"]), []).append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--known-scan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.run)
    sources = SOURCES
    if not rows:
        parser.error("No completed cells with metrics in the supplied runs")
    args.output.mkdir(parents=True, exist_ok=False)

    groups = grouped(rows)
    lines = ["# Natural CounterFact path patching (R07 extended relations)", "",
             "Fresh Slurm measurements on frozen pretrained LMs loaded from the local weight cache "
             "at pinned revisions. Each cell is one (model, relation, seed); the ratio is the maximum "
             "edge/module ratio over the candidate layer grid within that seed, with modules below "
             "1 pp excluded. Known facts are those the model answers greedily on every target token.", "",
             "| Model | Relation | Author bucket | Seeds | Known facts | Pairs | Max edge/module % | Edge |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for (model, relation), entries in sorted(groups.items(), key=lambda kv: (kv[0][0], AUTHOR_BUCKETS.get(kv[0][1], ""), kv[0][1])):
        values = [e["max_edge_module_pct"] for e in entries if e["max_edge_module_pct"] is not None]
        if values:
            mean, sd = mean_sd(values)
            text = f"{mean:.2f} ± {sd:.2f}" if sd is not None else f"{mean:.2f} (n=1)"
        else:
            text = "no eligible module"
        first = entries[0]
        edge = f"L{first['source']}→L{first['receiver']}" if first["source"] is not None else "-"
        lines.append(f"| {model} | {relation} | {first['bucket']} | {len(entries)} | {first['known_facts']} | "
                     f"{first['pairs']} | {text} | {edge} |")

    lines += ["", "## Bucket ranking per model", "",
              f"Rebuttal claim: {REBUTTAL_CLAIM}.", "",
              "| Model | Subject | Answer | Independent | Answer highest? |", "|---|---:|---:|---:|---|"]
    for model in sorted({row["model"] for row in rows}):
        best = {}
        for (m, relation), entries in groups.items():
            if m != model:
                continue
            values = [e["max_edge_module_pct"] for e in entries if e["max_edge_module_pct"] is not None]
            if not values:
                continue
            bucket = AUTHOR_BUCKETS.get(relation, "")
            mean = sum(values) / len(values)
            if bucket not in best or mean > best[bucket][1]:
                best[bucket] = (relation, mean)
        cells = []
        for bucket in ("subject", "answer", "independent"):
            cells.append(f"{best[bucket][1]:.2f} ({best[bucket][0]})" if bucket in best else "missing")
        answer_highest = ("undetermined" if "answer" not in best or len(best) < 3
                          else ("yes" if best["answer"][1] == max(v[1] for v in best.values()) else "no"))
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {answer_highest} |")

    if args.known_scan:
        scans = sorted(Path(args.known_scan).glob("cells/*/known_scan.json"))
        if scans:
            records = [read(path) for path in scans]
            relations = sorted({entry["relation"] for record in records for entry in record["relations"]})
            models = [record["model"].split("/")[-1] for record in records]
            lines += ["", "## Known facts per relation and model", "",
                      "Measured by `known_scan_only` runs. A relation a model does not know cannot be paired, "
                      "so these counts bound every cell above.", "",
                      "| Relation | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
            table = {record["model"].split("/")[-1]: {entry["relation"]: entry["known_facts"]
                                                      for entry in record["relations"]} for record in records}
            for relation in sorted(relations, key=lambda r: -sum(table[m].get(r, 0) for m in models)):
                lines.append(f"| {relation} | " + " | ".join(str(table[m].get(relation, "-")) for m in models) + " |")

    lines += ["", "## Reading these numbers", "",
              "- The maximum runs over every candidate edge in the grid, so a cell with few pairs maximises "
              "over noise more readily than a cell with many. Pair counts here differ by more than a factor of ten "
              "between cells, because they are capped by how many facts each model knows. Ratios from thin cells "
              "are not comparable with ratios from dense ones without equalising the pair budget.",
              "- These are the rebuttal's six additional relations, not the accepted paper's P103/P30/P19 strata. "
              "They do not confirm or refute the paper's own table, which has not been rerun.",
              "- P1412 is a disclosed substitute for P102, which the official snapshot lacks. A P1412 result is "
              "evidence about P1412 only.",
              "- Missing (model, relation) combinations are cells that failed or were preempted, or whose model "
              "knows too few facts to pair. They are absent, not zero.", ""]

    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    # Per-source maxima, because which sources the sweep contains decides the
    # maximum: the paper names its source layer per cell, and at those layers the
    # measurements agree, while a wider sweep finds larger ratios at earlier sources.
    # The paper's own footnote says fixing the source at L15 produces an apparent
    # subject localization that its sweep does not reproduce.
    with (args.output / "per_source.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "relation", "seed", "source", "max_edge_module_pct", "receiver"])
        for (model, relation, seed), entry in sorted(sources.items()):
            for source, best in sorted(entry.items()):
                writer.writerow([model, relation, seed, source, f"{best[0]:.4f}", best[1]])

    fields = ["model", "relation", "bucket", "seed", "known_facts", "pairs", "chunks",
              "max_edge_module_pct", "source", "receiver", "module_effect_pp", "run"]
    with (args.output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["relation"], r["seed"])))
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
