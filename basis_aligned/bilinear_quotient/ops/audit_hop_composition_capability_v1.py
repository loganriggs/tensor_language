#!/usr/bin/env python3
"""CPU capability gate for an existing stronger checkpoint, no probe/causal claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY)]
import torch
import hop_data as data
from hop_ablate import load


def panels():
    out = {'iid': data.sample_docs(16, torch.Generator().manual_seed(9909))}
    for kind, seed in (('unique_queries', 9910), ('short_cycles', 9911)):
        g = torch.Generator().manual_seed(seed)
        perm = torch.rand(16, 24, generator=g).argsort(1)
        fmap = torch.empty_like(perm).scatter_(1, perm, perm.roll(-1, 1)) if kind == 'unique_queries' else perm
        order = torch.rand(16, 24, generator=g).argsort(1)
        bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
        if kind == 'unique_queries':
            keys = torch.rand(16, 96, generator=g).argsort(1)[:, :48]
            qe, qk = keys//4, keys%4
        else:
            qe = torch.randint(24, (16, 48), generator=g); qk = torch.randint(4, (16, 48), generator=g)
        qa = data._fpow(fmap, 3)[torch.arange(16)[:, None], qk, qe]
        query = torch.stack((torch.full_like(qe, 24), qe, 25+qk, qa), -1).flatten(1)
        out[kind] = torch.cat((bindings, query), 1), qa, qk
    tok, _, _ = data.sample_docs(64, torch.Generator().manual_seed(9912))
    tok = tok[:, :52].clone()
    fmap = torch.empty(64, 24, dtype=torch.long).scatter_(1, tok[:, :48:2], tok[:, 1:48:2])
    qk = (torch.arange(64)%4)[:, None]; qe = tok[:, 49:50]
    qa = data._fpow(fmap, 3)[torch.arange(64)[:, None], qk, qe]
    tok[:, 50:51] = 25+qk; tok[:, 51:52] = qa
    out['cold_first_query'] = tok, qa, qk
    return out


def main():
    torch.set_num_threads(2); os.chdir(ROOT)
    started = time.perf_counter(); datasets = panels(); result = {}
    with torch.inference_mode():
        for name in ('attn4-rms-seed0', 'attn-mlp-attn-rms-seed0'):
            model, cfg = load(name); model = model.double().eval()
            measured = {}
            for panel, (tokens, answers, hops) in datasets.items():
                qp = torch.arange(50, tokens.shape[1]-1, 4)
                logits = torch.cat([model(tokens[i:i+4, :-1])[:, qp] for i in range(0, len(tokens), 4)])
                correct = logits.argmax(-1) == answers
                probabilities = logits.softmax(-1).gather(-1, answers[..., None]).squeeze(-1)
                measured[panel] = {str(k): {'n': int((hops == k).sum()),
                                            'accuracy': float(correct[hops == k].double().mean()),
                                            'gold_probability': float(probabilities[hops == k].mean())} for k in range(4)}
            result[name] = {'checkpoint_sha256': hashlib.sha256((ROOT/f'runs_hop/{name}/model.pt').read_bytes()).hexdigest(),
                            'config': cfg, 'parameters': sum(p.numel() for p in model.parameters()), 'panels': measured}
    strong = result['attn4-rms-seed0']['panels']
    out = {'experiment': 'hop_composition_capability_v1', 'models': result,
           'predictions': {'strong_novel_higher_hops': all(strong[p][str(k)]['accuracy'] >= .8 for p in ('unique_queries', 'cold_first_query') for k in (2, 3)),
                           'short_cycle_higher_hops': all(strong['short_cycles'][str(k)]['accuracy'] >= .8 for k in (2, 3))},
           'panel_token_sha256': {p: hashlib.sha256(v[0].numpy().tobytes()).hexdigest() for p, v in datasets.items()},
           'scope': 'capability only; no mechanistic conclusion from prior linear probes',
           'wall_seconds': time.perf_counter()-started, 'gpu_seconds': 0}
    (POLY/'HOP_COMPOSITION_CAPABILITY_V1_RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
