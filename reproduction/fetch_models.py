"""Pre-download pretrained weights into the local Hub cache (stdlib only, login-safe).

The compute nodes should never need network access: every model required by
E09, E11, E12 and E14 is fetched here into ``reproduction/cache/huggingface/hub``
using the standard Hub cache layout, so ``HF_HOME`` plus ``HF_HUB_OFFLINE=1``
resolves them without contacting huggingface.co.

Each download records the resolved commit revision and per-file SHA-256.
Gated repositories use the ``HF_TOKEN`` environment
variable or ``~/.cache/huggingface/token``; the token value is never written to
the manifest or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "cache/huggingface/hub"
ENDPOINT = "https://huggingface.co"

# Paper pointers: Final_Paper/neurips_2026.tex Table tab:pythia-natural (E09),
# Appendix app:pythia-synth (E11) and the two-hop appendix (E14).
MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

# Weights plus everything needed to rebuild the tokenizer; other serialization
# formats (.bin duplicates, ONNX, GGUF, TensorFlow) are skipped deliberately.
KEEP_SUFFIXES = (".safetensors", ".json", ".model", ".txt")
SKIP_PREFIXES = ("onnx/", "openvino/", "coreml/", "tf_", "flax_", "original/")
SKIP_NAMES = ("README.md", ".gitattributes")


def token() -> str | None:
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value.strip()
    stored = Path.home() / ".cache/huggingface/token"
    return stored.read_text().strip() if stored.is_file() else None


def request(url: str, auth: str | None):
    headers = {"User-Agent": "reproduction-fetch-models/1"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    return urllib.request.Request(url, headers=headers)


def api_json(url: str, auth: str | None):
    with urllib.request.urlopen(request(url, auth), timeout=60) as response:
        return json.load(response)


def wanted(path: str) -> bool:
    if path in SKIP_NAMES or any(path.startswith(p) for p in SKIP_PREFIXES):
        return False
    return path.endswith(KEEP_SUFFIXES)


def listing(repo: str, auth: str | None):
    """Resolve the current main commit and the files to fetch from it."""
    info = api_json(f"{ENDPOINT}/api/models/{repo}", auth)
    revision = info["sha"]
    tree = api_json(f"{ENDPOINT}/api/models/{repo}/tree/{revision}?recursive=1", auth)
    files = [entry for entry in tree if entry["type"] == "file" and wanted(entry["path"])]
    if not any(entry["path"].endswith(".safetensors") for entry in files):
        raise RuntimeError(f"{repo}: no safetensors weights in revision {revision}")
    return revision, files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo: str, revision: str, entry: dict, snapshot: Path, auth: str | None) -> dict:
    target = snapshot / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (entry.get("lfs") or {}).get("oid")
    if target.is_file() and target.stat().st_size == entry["size"]:
        local = sha256(target)
        if expected is None or local == expected:
            return {"path": entry["path"], "size": entry["size"], "sha256": local, "fetched": False}
    url = f"{ENDPOINT}/{repo}/resolve/{revision}/{entry['path']}"
    partial = target.with_suffix(target.suffix + ".incomplete")
    with urllib.request.urlopen(request(url, auth), timeout=120) as response, partial.open("wb") as out:
        while block := response.read(1 << 22):
            out.write(block)
    local = sha256(partial)
    if expected is not None and local != expected:
        partial.unlink()
        raise RuntimeError(f"{repo}/{entry['path']}: sha256 {local} != advertised {expected}")
    if partial.stat().st_size != entry["size"]:
        partial.unlink()
        raise RuntimeError(f"{repo}/{entry['path']}: size mismatch")
    partial.replace(target)
    return {"path": entry["path"], "size": entry["size"], "sha256": local, "fetched": True}


def fetch(name: str, repo: str, cache: Path, auth: str | None) -> dict:
    revision, files = listing(repo, auth)
    folder = cache / ("models--" + repo.replace("/", "--"))
    snapshot = folder / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    records = [download(repo, revision, entry, snapshot, auth) for entry in files]
    refs = folder / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision)
    manifest = {"model": name, "repo": repo, "revision": revision,
                "snapshot": str(snapshot), "files": records,
                "bytes": sum(record["size"] for record in records)}
    (folder / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def cached_revisions(cache: Path = DEFAULT_CACHE) -> dict:
    """Repository id to the snapshot revision this cache actually holds.

    Configurations pin these revisions to select the cached weights.
    """
    found = {}
    for manifest in sorted(cache.glob("models--*/download_manifest.json")):
        record = json.loads(manifest.read_text())
        found[record["repo"]] = record["revision"]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=sorted(MODELS), help="default: all")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--list", action="store_true", help="resolve revisions and sizes without downloading")
    args = parser.parse_args()
    auth = token()
    selected = args.model or sorted(MODELS)
    args.cache.mkdir(parents=True, exist_ok=True)
    for name in selected:
        repo = MODELS[name]
        try:
            if args.list:
                revision, files = listing(repo, auth)
                total = sum(entry["size"] for entry in files)
                print(f"{name}\t{repo}\t{revision}\t{len(files)} files\t{total / 2**30:.2f} GiB", flush=True)
                continue
            manifest = fetch(name, repo, args.cache, auth)
            print(f"{name}\t{manifest['revision']}\t{manifest['bytes'] / 2**30:.2f} GiB\t{manifest['snapshot']}",
                  flush=True)
        except urllib.error.HTTPError as error:
            gated = " (gated repository: accept its license with this account)" if error.code in (401, 403) else ""
            print(f"{name}\tHTTP {error.code}{gated}", flush=True)
        except (urllib.error.URLError, RuntimeError) as error:
            print(f"{name}\tFAILED {error}", flush=True)


if __name__ == "__main__":
    main()
