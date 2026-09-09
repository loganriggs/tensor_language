#!/usr/bin/env python3
"""Counterbalanced causal suffix-join source hypothesis,96 fresh worlds.

pred_a_instrument: all full outputs/centered cuts/joint identities1e-9.
pred_b_causal_join_source: each12 positive groups acc>=.8,J loss>=.5,abs I loss<=.1.
pred_c_join_source_sufficient: keepJQ query KL mean<=1e-3,p99<=1e-2.
pred_d_fixed_point_failure: native fixed-point hop3 mean gold P<=.15.
Null closes fixed later-suffix source rule. All parameters charged; no join-head claim.
384 correlated orders+32 fixedpoint controls,B4 FP64,1800s,<256MiB tensor; managed GPU.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_causal_join_source pred_c_join_source_sufficient pred_d_fixed_point_failure
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__)
OUT = POLY/'CAUSAL_SUFFIX_JOIN_V1_RESULT.json'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'causal_suffix_join_reference.py', POLY/'cold_composition_source_reference.py', POLY/'contextual_history_reference.py',
         POLY/'CAUSAL_SUFFIX_JOIN_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = 'e35bd2e73e7ce3a87c09a838b8172d35eafead2e239ea70312a80695af981aa6'
ARMS = ('full', 'removeJ', 'removeI', 'removeJI', 'keepJQ')


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def run(torch, R, score, controls):
    import hop_data
    from hop_ablate import load
    signal.alarm(1800); started = time.perf_counter(); os.chdir(ROOT)
    assert not OUT.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0')
    model = model.to(device='cuda', dtype=torch.float64).eval()
    program = R.HistoryReadout(model).eval()
    results = {}
    with torch.inference_mode():
        for pop, (tokens, answers, orders, worlds) in R.populations(torch).items():
            native = {a: [] for a in ARMS}; candidate = {a: [] for a in ARMS}
            exact = True; max_error = 0.; source_counts = []
            for i in range(0, len(tokens), 4):
                masks = {k: v.cuda() for k, v in R.source_masks(tokens[i:i+4]).items()}
                source_counts.extend(torch.stack([v.sum((1, 2)) for v in masks.values()], -1).cpu().tolist())
                tok = tokens[i:i+4].cuda()
                parts = program.parts(tok, masks)
                full = parts['R']+parts['H']+parts['B']+parts['O']
                outputs = {'full': full, 'removeJ': full-parts['H'], 'removeI': full-parts['B'],
                           'removeJI': full-parts['H']-parts['B'], 'keepJQ': full.clone()}
                outputs['keepJQ'][:, 50] = (parts['R']+parts['H']+parts['C'])[:, 50]
                other = torch.zeros_like(masks['H']); other[:, 50] = ~(masks['H'] | masks['C'])[:, 50]
                cuts = {'full': None, 'removeJ': masks['H'], 'removeI': masks['B'],
                        'removeJI': masks['H'] | masks['B'], 'keepJQ': other}
                for a in ARMS:
                    n = R.native_forward(model, tok, cuts[a]); c = outputs[a]
                    exact &= bool(torch.allclose(n, c, atol=1e-9, rtol=1e-9))
                    max_error = max(max_error, float((n-c).abs().max()))
                    native[a].append(n.cpu()); candidate[a].append(c.cpu())
            native = {a: torch.cat(v) for a, v in native.items()}; candidate = {a: torch.cat(v) for a, v in candidate.items()}
            probs = {a: v[:, 50].softmax(-1).gather(-1, answers[:, None]).squeeze(-1) for a, v in native.items()}
            groups = {}
            for k in range(1 if pop == 'fixed_point' else 6):
                select = orders == k
                groups[str(k)] = {'n_worlds': int(select.sum()),
                    'arms': {a: {'accuracy': float((v[:, 50].argmax(-1) == answers)[select].double().mean()),
                                 'gold_probability': float(probs[a][select].mean()),
                                 'gold_probability_loss': float((probs['full']-probs[a])[select].mean())} for a, v in native.items()},
                    'keepJQ_distribution': score.distribution(candidate['keepJQ'][select, 50], native['full'][select, 50], torch)}
            def center(x):
                return x-x.mean(-1, keepdim=True)
            effect_close = all(torch.allclose(center(candidate[a]-candidate['full']), center(native[a]-native['full']), atol=1e-9, rtol=1e-9) for a in ARMS)
            joint = native['removeJ']+native['removeI']-native['full']
            joint_close = bool(torch.allclose(joint, native['removeJI'], atol=1e-9, rtol=1e-9))
            results[pop] = {'groups': groups, 'exact': exact, 'max_abs_replay_error': max_error,
                            'effect_vectors_close': effect_close, 'joint_close': joint_close,
                            'joint_max_abs': float((joint-native['removeJI']).abs().max()), 'source_counts_T_P_Q': source_counts,
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                            'rows': [{'world': int(worlds[i]), 'order': int(orders[i]), 'answer': int(answers[i]),
                                      'query_logits': {a: native[a][i, 50].tolist() for a in ARMS}} for i in range(len(tokens))]}
            print(json.dumps({'population': pop, 'groups': groups, 'replay_max_abs': max_error}), flush=True)
    pred_a = controls['passed'] and all(r['exact'] and r['effect_vectors_close'] and r['joint_close'] for r in results.values())
    positive = [r for p, r in results.items() if p != 'fixed_point']
    pred_b = pred_a and all(g['arms']['full']['accuracy'] >= .8 and g['arms']['removeJ']['gold_probability_loss'] >= .5
                            and abs(g['arms']['removeI']['gold_probability_loss']) <= .1 for r in positive for g in r['groups'].values())
    pred_c = pred_a and all(g['keepJQ_distribution']['passed'] for r in positive for g in r['groups'].values())
    pred_d = pred_a and results['fixed_point']['groups']['0']['arms']['full']['gold_probability'] <= .15
    receipt = {'experiment': 'causal_suffix_join_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': controls, 'populations': results,
               'independent_worlds': 96, 'positive_order_variants': 384, 'fixed_point_controls': 32,
               'orders': [list(o) for o in R.ORDERS], 'chain_binding_slots': list(R.SLOTS),
               'compiled_opaque_constants': sum(p.numel() for p in program.parameters())+program.folded.numel(),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_causal_join_source': bool(pred_b),
                               'pred_c_join_source_sufficient': bool(pred_c), 'pred_d_fixed_point_failure': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('causal_join_source_nominated' if pred_b else 'later_suffix_source_rule_rejected'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import causal_suffix_join_reference as R
    import cold_composition_source_reference as old_reference
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls(); old_checks = old_reference.controls()
    checks['checks'].update({'fold_'+k: v for k, v in old_checks['checks'].items()})
    checks['passed'] = all(checks['checks'].values()); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
