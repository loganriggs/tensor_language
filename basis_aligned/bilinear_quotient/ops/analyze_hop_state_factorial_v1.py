#!/usr/bin/env python3
"""CPU post-null diagnostic; reuse opened panels/books, never refit or promote."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY)]
import torch
import hop_data
from hop_ablate import load
import hop_state_reference as R
import run_hop_state_compiler_v1 as parent
import run_equality_router_v1 as score


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    os.chdir(ROOT)
    result = json.loads(parent.OUT.read_text())
    assert result['terminal'] == 'fixed_hop_state_rejected'
    assert score.digest(parent.BOOKS) == result['books_sha256']
    assert parent.digest_bound() == parent.EXPECTED
    data = torch.load(parent.BOOKS, map_location='cpu', weights_only=False)
    model, _ = load('attn-mlp-attn-rms-seed0')
    model = model.double().eval()
    programs = {}
    for arm, replace in {'Q': ('q1',), 'K': ('k1',), 'QK': ('q1', 'k1')}.items():
        prog = R.HopStateReadout(model, data['books'], data['counts'] > 0)
        for name in ('q1', 'k1'):
            if name not in replace:
                setattr(prog.background.layers[-1], name, copy.deepcopy(getattr(model.layers[-1], name)))
        programs[arm] = prog.eval()
    panels = {}
    with torch.inference_mode():
        for pop, tokens in parent.populations(torch, hop_data).items():
            arrays = {a: [] for a in ('base', 'Q', 'K', 'QK')}
            for i in range(0, len(tokens), 4):
                tok = tokens[i:i+4, :-1]
                arrays['base'].append(model(tok))
                for a, prog in programs.items():
                    arrays[a].append(R.full(prog, tok))
            arrays = {a: torch.cat(v) for a, v in arrays.items()}
            qp = torch.arange(50, tokens.shape[1]-1, 4)
            def centered(x):
                return x-x.mean(-1, keepdim=True)
            base = arrays['base']
            interaction = centered(arrays['QK']-arrays['Q']-arrays['K']+base)
            total = centered(arrays['QK']-base)
            effects = {a: centered(v-base) for a, v in arrays.items() if a != 'base'}
            panels[pop] = {
                'distribution': {a: {'all': score.distribution(v, base, torch),
                                      'query': score.distribution(v[:, qp], base[:, qp], torch)}
                                 for a, v in arrays.items() if a != 'base'},
                'centered_error_rms': {a: float(v.square().mean().sqrt()) for a, v in effects.items()},
                'interaction_rms': float(interaction.square().mean().sqrt()),
                'interaction_over_joint_error': float(interaction.norm()/total.norm().clamp_min(1e-30)),
                'cpu_parent_query_kl_difference': abs(score.distribution(arrays['QK'][:, qp], base[:, qp], torch)['mean_kl']-result['populations'][pop]['distribution']['native']['query']['mean_kl'])}
    pred_q = all(v['passed'] for p in panels.values() for v in p['distribution']['Q'].values())
    pred_k = all(v['passed'] for p in panels.values() for v in p['distribution']['K'].values())
    pred_i = all(p['interaction_over_joint_error'] <= .1 for p in panels.values())
    out = {'experiment': 'hop_state_factorial_v1', 'scope': 'post-null diagnostic on already opened populations; no held-out circuit claim',
           'parent_sha256': score.digest(parent.OUT), 'books_sha256': result['books_sha256'],
           'predictions': {'query_code_alone_sufficient': pred_q, 'key_code_alone_sufficient': pred_k,
                           'interaction_below_tenth_of_joint_error': pred_i},
           'populations': panels, 'wall_seconds': time.perf_counter()-started, 'gpu_seconds': 0}
    (POLY/'HOP_STATE_FACTORIAL_V1_RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
