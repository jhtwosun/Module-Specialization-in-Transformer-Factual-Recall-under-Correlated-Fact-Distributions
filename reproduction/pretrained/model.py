"""Native HF forward with reversible residual hooks; zero-based block indices."""
from contextlib import contextmanager
import torch
from torch import nn
from transformers import AutoModelForCausalLM, GPTNeoXConfig, LlamaConfig


class Instrumented(nn.Module):
    def __init__(self, c, classes=None):
        super().__init__()
        dtype = getattr(torch, c.get('dtype', 'float32' if classes else 'bfloat16'))
        if c.get('random_init', False):
            cls = LlamaConfig if c.get('architecture') == 'llama' else GPTNeoXConfig
            config = cls(vocab_size=512, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128, attention_dropout=0.0, hidden_dropout=0.0)
            config._attn_implementation = 'eager'
            self.base = AutoModelForCausalLM.from_config(config).to(dtype=dtype)
        else:
            self.base = AutoModelForCausalLM.from_pretrained(
                c['model'], revision=c.get('revision', 'main'), torch_dtype=dtype,
                attn_implementation='eager', local_files_only=c.get('local_files_only', False))
        self.body = self.base.gpt_neox if hasattr(self.base, 'gpt_neox') else self.base.model
        self.layers = self.body.layers
        self.head = nn.Linear(self.base.config.hidden_size, classes).to(dtype=dtype) if classes else None
        self.probes = nn.ModuleList([nn.Linear(self.base.config.hidden_size, classes).to(dtype=dtype)
                                    for _ in range(len(self.layers) + 1)]) if classes else None
        self.base.config.use_cache = False
        if classes:
            for p in self.base.get_output_embeddings().parameters():
                p.requires_grad_(False)

    @contextmanager
    def hooks(self, cache, module=None, edge=None):
        handles = []
        def pre(i):
            def hook(layer, args):
                h = args[0]
                if module is not None and module[0] == -1 and i == 0:
                    h = h.clone()
                    h[:, -1] = module[1]
                if edge is not None and edge[0] == i:
                    h = h.clone()
                    h[:, -1] += edge[1]
                cache['in'][i] = h[:, -1].clone()
                return (h,) + args[1:]
            return hook
        def post(i):
            def hook(layer, args, out):
                h = out[0] if isinstance(out, tuple) else out
                if module is not None and module[0] == i:
                    h = h.clone()
                    h[:, -1] = module[1]
                cache['out'][i] = h[:, -1].clone()
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        try:
            for i, layer in enumerate(self.layers):
                handles.extend([layer.register_forward_pre_hook(pre(i)), layer.register_forward_hook(post(i))])
            yield
        finally:
            for h in handles:
                h.remove()

    def forward(self, x, mask=None, module=None, edge=None):
        cache = {'in': {}, 'out': {}}
        with self.hooks(cache, module, edge):
            # Calling the native backbone retains its causal attention mask and rotary handling.
            out = self.body(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True)
        h = out.last_hidden_state[:, -1]
        logits = self.head(h) if self.head is not None else self.base.get_output_embeddings()(h)
        return logits, cache

    def probe_logits(self, cache):
        states = [cache['in'][0]] + [cache['out'][i] for i in range(len(self.layers))]
        return [p(h.detach()) for p, h in zip(self.probes, states)]
