"""Reevaluate completed pilot checkpoints; no retraining, Slurm GPU only."""
import argparse
import hashlib
import json
from pathlib import Path
from .run import guard, dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    guard()
    import torch
    from .model import Transformer
    from .evaluate import evaluate
    from .tests_gpu import main as invariants
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    invariants()
    torch.set_num_threads(4)
    manifest = json.loads((a.run/'manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    for cell in manifest['cells']:
        src = a.run/cell['output']
        cfg = json.loads((src/'config.json').read_text())
        checkpoint = src/'checkpoint.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['step'] != cfg['steps'] or state['config'] != cfg:
            raise ValueError('Checkpoint incomplete or configuration mismatch')
        model = Transformer(cfg).cuda()
        model.load_state_dict(state['model'])
        data = torch.load(src/'facts.pt', map_location='cpu', weights_only=False)
        out = a.output/cell['cell']
        out.mkdir()
        dump(out/'config.json', cfg)
        dump(out/'provenance.json', dict(parent_run=str(a.run.resolve()), cell=cell['cell'],
             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
             torch=torch.__version__, gpu=torch.cuda.get_device_name(),
             purpose='Same trained checkpoint; query-only and recovered all-position ablation separately'))
        evaluate(model, data['table'], data['classes'], cfg, out)
        print('REEVALUATED', cell['cell'], flush=True)
        del model
    print('QUERY_AUDIT_OK', flush=True)


if __name__ == '__main__':
    main()
