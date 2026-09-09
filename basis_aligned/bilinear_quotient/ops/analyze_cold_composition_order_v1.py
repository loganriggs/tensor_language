#!/usr/bin/env python3
"""CPU, opened-panel diagnostic: causal visibility of answer facts at earlier sources."""
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
import hop_data
from hop_ablate import load
import cold_composition_source_reference as R
import run_cold_composition_source_v1 as parent


def main():
    torch.set_num_threads(2); os.chdir(ROOT); started = time.perf_counter()
    old = json.loads(parent.OUT.read_text())
    assert old['terminal'] == 'direct_answer_binding_not_supported'
    assert parent.bound_hash() == parent.EXPECTED
    assert hashlib.sha256(parent.CHECKPOINT.read_bytes()).hexdigest() == parent.EXPECTED_CHECKPOINT
    model, _ = load('attn4-rms-seed0'); model = model.double().eval()
    program = R.HistoryReadout(model).eval(); panels = {}; rows = []
    with torch.inference_mode():
        for pop, (all_tokens, all_answers, all_hops) in parent.populations(torch, hop_data).items():
            tokens = all_tokens[all_hops == 3]; answers = all_answers[all_hops == 3]
            pop_rows = []
            for i in range(0, len(tokens), 4):
                tok = tokens[i:i+4]; masks = R.source_masks(tok)
                parts = program.parts(tok, masks)
                full = parts['R']+parts['H']+parts['B']+parts['O']
                assert torch.allclose(full, model(tok), atol=1e-9, rtol=1e-9)
                logits = {'full': full[:, 50], 'removeT': (full-parts['H'])[:, 50],
                          'removeP': (full-parts['B'])[:, 50], 'removeTP': (full-parts['H']-parts['B'])[:, 50]}
                for j, token in enumerate(tok):
                    target = torch.where(masks['H'][j, 50])[0].tolist()
                    prior = torch.where(masks['B'][j, 50])[0].tolist()
                    category = 'no_distinct_prefix' if not prior else ('target_latest' if max(target) > max(prior) else 'prefix_latest')
                    keys, vals = token[:48:2].tolist(), token[1:48:2].tolist()
                    fmap = dict(zip(keys, vals)); positions = {e: 2*k+1 for k, e in enumerate(keys)}
                    current = int(token[49]); chain = []
                    for _ in range(3):
                        chain.append(positions[current]); current = fmap[current]
                    answer = int(answers[i+j]); prob = {a: float(v[j].softmax(-1)[answer]) for a, v in logits.items()}
                    row = {'population': pop, 'world': i+j, 'category': category, 'chain_value_positions': chain,
                           'answer': answer, 'native_correct': int(logits['full'][j].argmax()) == answer,
                           'gold_probability': prob, 'loss_T': prob['full']-prob['removeT'], 'loss_P': prob['full']-prob['removeP'],
                           'query_logits': {a: v[j].tolist() for a, v in logits.items()}}
                    pop_rows.append(row); rows.append(row)
            groups = {}
            for category in ('target_latest', 'prefix_latest', 'no_distinct_prefix'):
                subset = [r for r in pop_rows if r['category'] == category]
                if subset:
                    groups[category] = {'n': len(subset), 'native_accuracy': sum(r['native_correct'] for r in subset)/len(subset),
                                        'mean_loss_T': sum(r['loss_T'] for r in subset)/len(subset),
                                        'mean_loss_P': sum(r['loss_P'] for r in subset)/len(subset)}
            orders = {}
            for row in pop_rows:
                chain = row['chain_value_positions']
                order = ','.join(str(j+1) for j in sorted(range(3), key=lambda j: chain[j]))
                if len(set(chain)) == 3:
                    orders.setdefault(order, []).append(row)
            panels[pop] = {'visibility_groups': groups,
                           'distinct_chain_order_groups': {k: {'n': len(v), 'mean_loss_T': sum(r['loss_T'] for r in v)/len(v),
                                                               'mean_loss_P': sum(r['loss_P'] for r in v)/len(v)} for k, v in orders.items()}}
    a = all(abs(p['visibility_groups']['target_latest']['mean_loss_P']) <= .1 for p in panels.values())
    b = all(p['visibility_groups']['prefix_latest']['mean_loss_P'] >= .25 and
            p['visibility_groups']['prefix_latest']['mean_loss_P']-p['visibility_groups']['target_latest']['mean_loss_P'] >= .2 for p in panels.values())
    out = {'experiment': 'cold_composition_order_v1', 'scope': 'diagnostic reuse of opened parent panels; not independent confirmation',
           'parent_sha256': hashlib.sha256(parent.OUT.read_bytes()).hexdigest(), 'populations': panels,
           'predictions': {'earlier_source_inert_when_target_future': a, 'earlier_source_effect_tracks_fact_visibility': b},
           'rows': rows, 'wall_seconds': time.perf_counter()-started, 'gpu_seconds': 0}
    (POLY/'COLD_COMPOSITION_ORDER_V1_RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: out[k] for k in ('predictions', 'populations', 'wall_seconds')}))


if __name__ == '__main__':
    main()
