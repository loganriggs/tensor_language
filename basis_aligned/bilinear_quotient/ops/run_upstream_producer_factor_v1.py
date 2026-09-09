#!/usr/bin/env python3
"""Exact normalized producer-to-reader atlas, fresh source-conditioned outcomes.

pred_a_instrument: exact producer/reader/full replay1e-9, native repeated p>=.8.
pred_b_selective_path: same one of15 paths loses>=.25 repeated high-hop gold P,
absolute novel-hop1 change<=.10 in both populations; no selected-row filtering.
pred_c_additive_joint: sum single reader logit effects predicts joint<=.01 relative.
pred_d_live_edges: all15 native reader cuts have centered RMS>1e-6.
Null: dependencies distributed or interacting, retain the complete product.
128docs x19arms,B4,FP64,1800s guard,<256MiB each analysis tensor. No fitting.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selective_path pred_c_additive_joint pred_d_live_edges
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
OUT = POLY/'UPSTREAM_PRODUCER_FACTOR_V1_RESULT.json'
BOUND = [POLY/'upstream_producer_reference.py', POLY/'contextual_history_reference.py',
         POLY/'UPSTREAM_PRODUCER_FACTOR_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '30b7f28ba71fc227969331492f8f11445b4c25ac885d994e39cb0c5919cde674'


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch, data):
    panels = {'iid': data.sample_docs(64, torch.Generator().manual_seed(7909))}
    g = torch.Generator().manual_seed(7910)
    fmap = torch.rand(64, 24, generator=g).argsort(1)
    order = torch.rand(64, 24, generator=g).argsort(1)
    bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
    qe = torch.randint(24, (64, 48), generator=g)
    qk = torch.randint(4, (64, 48), generator=g)
    qa = data._fpow(fmap, 3)[torch.arange(64)[:, None], qk, qe]
    blocks = torch.stack((torch.full_like(qe, 24), qe, 25+qk, qa), -1).flatten(1)
    panels['ood_short_cycles'] = (torch.cat((bindings, blocks), 1), qa, qk)
    return panels


def run(torch, R, H, score, controls):
    from hop_ablate import load
    import hop_data
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    assert not OUT.exists()
    assert score.digest(score.CHECKPOINT) == score.EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn-mlp-attn-rms-seed0')
    model = model.to(device='cuda', dtype=torch.float64).eval()
    arms = {'native': ()}
    arms.update({f'{p}_{r}': ((p, r),) for p in R.PRODUCERS for r in R.READERS})
    arms.update({f'{p}_joint': tuple((p, r) for r in R.READERS) for p in R.PRODUCERS})
    results = {}
    with torch.inference_mode():
        for pop, (tokens, answers, hops) in populations(torch, hop_data).items():
            arrays = {a: [] for a in arms}; repeated = []
            exact = True; max_error = 0.
            for i in range(0, len(tokens), 4):
                _, rep, _, _ = H.source_masks(tokens[i:i+4]); repeated.append(rep)
                tok = tokens[i:i+4, :-1].cuda()
                x, parts = R.producers(model, tok)
                layer = model.layers[-1]
                norm = layer.norm(x)
                for reader in R.READERS:
                    projection = getattr(layer, reader)
                    rec = sum(projection(v) for v in parts.values())
                    expected = projection(norm)
                    exact &= bool(torch.allclose(rec, expected, atol=1e-9, rtol=1e-9))
                    max_error = max(max_error, float((rec-expected).abs().max()))
                original = model(tok)
                for arm, cuts in arms.items():
                    out = R.forward(model, tok, cuts)
                    arrays[arm].append(out.cpu())
                    if arm == 'native':
                        exact &= bool(torch.allclose(out, original, atol=1e-9, rtol=1e-9))
                        max_error = max(max_error, float((out-original).abs().max()))
            arrays = {a: torch.cat(v) for a, v in arrays.items()}
            rep = torch.cat(repeated)
            groups = {'high_repeated': (hops >= 2)&rep, 'high_novel': (hops >= 2)&~rep,
                      'hop1_novel': (hops == 1)&~rep}
            query = {a: v[:, hop_data.ANS_POS] for a, v in arrays.items()}
            probabilities = {a: v.softmax(-1).gather(-1, answers[..., None]).squeeze(-1) for a, v in query.items()}
            measured = {}
            for name, select in groups.items():
                assert bool(select.any())
                measured[name] = {'n': int(select.sum()), 'native_gold_probability': float(probabilities['native'][select].mean()),
                                  'native_accuracy': float((query['native'].argmax(-1) == answers)[select].double().mean()),
                                  'arms': {a: {'gold_probability_loss': float((probabilities['native']-probabilities[a])[select].mean()),
                                               'accuracy': float((v.argmax(-1) == answers)[select].double().mean())} for a, v in query.items()}}
            def center(x):
                return x-x.mean(-1, keepdim=True)
            effects = {a: center(v-arrays['native']) for a, v in arrays.items() if a != 'native'}
            joints = {p: score.effect_error(sum(effects[f'{p}_{r}'] for r in R.READERS), effects[f'{p}_joint'], torch) for p in R.PRODUCERS}
            passing = [a for a in effects if not a.endswith('_joint')
                       and measured['high_repeated']['arms'][a]['gold_probability_loss'] >= .25
                       and abs(measured['hop1_novel']['arms'][a]['gold_probability_loss']) <= .10]
            results[pop] = {'groups': measured, 'passing_paths': passing, 'joint_additivity': joints,
                            'centered_effect_rms': {a: float(v.square().mean().sqrt()) for a, v in effects.items()},
                            'full_distribution_changes': {a: score.distribution(v, arrays['native'], torch) for a, v in arrays.items()},
                            'exact': exact, 'max_reassembly_error': max_error,
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            print(json.dumps({'population': pop, 'passing_paths': passing, 'joints': joints,
                              'native_repeat_probability': measured['high_repeated']['native_gold_probability']}), flush=True)
    common = sorted(set.intersection(*(set(r['passing_paths']) for r in results.values())))
    pred_a = controls['passed'] and all(r['exact'] and r['groups']['high_repeated']['native_gold_probability'] >= .8 for r in results.values())
    pred_b = pred_a and bool(common)
    pred_c = pred_a and all(v['passed'] for r in results.values() for v in r['joint_additivity'].values())
    pred_d = all(v > 1e-6 for r in results.values() for a, v in r['centered_effect_rms'].items() if not a.endswith('_joint'))
    receipt = {'experiment': 'upstream_producer_factor_v1', 'runner_sha256': score.digest(SOURCE),
               'bound_sha256': EXPECTED, 'checkpoint_sha256': score.EXPECTED_CHECKPOINT, 'controls': controls,
               'scope': 'normalized reader-edge cuts with native live RMS gain; not upstream state removal',
               'populations': results, 'common_selective_paths': common, 'tested_single_paths': 15,
               'native_opaque_parameters_retained': sum(p.numel() for p in model.parameters()),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_selective_path': bool(pred_b),
                               'pred_c_additive_joint': bool(pred_c), 'pred_d_live_edges': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('selective_reader_paths_nominated' if pred_b else 'no_selective_single_reader_path'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'common_selective_paths', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import upstream_producer_reference as R
    import contextual_history_reference as H
    import run_equality_router_v1 as score
    torch.set_num_threads(2)
    checks = R.controls()
    assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False}))
        return
    run(torch, R, H, score, checks)


if __name__ == '__main__':
    main()
