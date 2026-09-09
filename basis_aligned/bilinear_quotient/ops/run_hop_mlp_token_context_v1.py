#!/usr/bin/env python3
"""Known bilinear terms, new shared hop-lookup sufficiency and intervention test.

pred_a_instrument: all8 native/candidate outputs1e-9; native repeat probability>=.8.
pred_b_mixed_sufficiency: X-only full/query KL mean<=1e-3,p99<=1e-2 both populations.
pred_c_shared_mixed_dependency: removeX gold-prob loss>=.25 on repeats and novelhop1;
removeTT/removeCC absolute probability change<=.10, both populations.
pred_d_joint_interventions: all8 centered native/compiled intervention vectors1e-9.
Null: distributed MLP terms; no branch/rank/feature sweep or compression claim.
128docs x8arms x2executors,B4 FP64,1800s,<256MiB each tensor; managed GPU only.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_mixed_sufficiency pred_c_shared_mixed_dependency pred_d_joint_interventions
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
OUT = POLY/'HOP_MLP_TOKEN_CONTEXT_V1_RESULT.json'
BOUND = [POLY/'hop_mlp_terms_reference.py', POLY/'contextual_history_reference.py',
         POLY/'HOP_MLP_TOKEN_CONTEXT_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '6c80d06a6b7a32b41036d844aa48237c7ddc6daa3a65f5f44d1d809ec8151d7d'


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch, data):
    panels = {'iid': data.sample_docs(64, torch.Generator().manual_seed(8909))}
    g = torch.Generator().manual_seed(8910)
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
    results = {}
    with torch.inference_mode():
        for pop, (tokens, answers, hops) in populations(torch, hop_data).items():
            native = {a: [] for a in R.ARMS}; candidate = {a: [] for a in R.ARMS}; repeated = []
            exact = True; effect_close = True; max_error = 0.
            for i in range(0, len(tokens), 4):
                _, rep, _, _ = H.source_masks(tokens[i:i+4]); repeated.append(rep)
                tok = tokens[i:i+4, :-1].cuda()
                for arm, keep in R.ARMS.items():
                    n = R.native_cut(model, tok, keep)
                    c = R.forward(model, tok, keep)
                    exact &= bool(torch.allclose(n, c, atol=1e-9, rtol=1e-9))
                    max_error = max(max_error, float((n-c).abs().max()))
                    native[arm].append(n.cpu()); candidate[arm].append(c.cpu())
            native = {a: torch.cat(v) for a, v in native.items()}
            candidate = {a: torch.cat(v) for a, v in candidate.items()}
            rep = torch.cat(repeated)
            query = {a: v[:, hop_data.ANS_POS] for a, v in native.items()}
            probs = {a: v.softmax(-1).gather(-1, answers[..., None]).squeeze(-1) for a, v in query.items()}
            groups = {}
            for name, select in {'high_repeated': (hops >= 2)&rep, 'high_novel': (hops >= 2)&~rep,
                                  'hop1_novel': (hops == 1)&~rep}.items():
                assert bool(select.any())
                groups[name] = {'n': int(select.sum()), 'native_gold_probability': float(probs['full'][select].mean()),
                                 'arms': {a: {'gold_probability': float(probs[a][select].mean()),
                                              'gold_probability_loss': float((probs['full']-probs[a])[select].mean()),
                                              'accuracy': float((v.argmax(-1) == answers)[select].double().mean())} for a, v in query.items()}}
            def center(x):
                return x-x.mean(-1, keepdim=True)
            effects = {}
            for a in R.ARMS:
                nc = center(native[a]-native['full']); cc = center(candidate[a]-candidate['full'])
                effect_close &= bool(torch.allclose(nc, cc, atol=1e-9, rtol=1e-9))
                effects[a] = score.effect_error(cc, nc, torch)
            distributions = {a: {'all': score.distribution(candidate[a], native['full'], torch),
                                  'query': score.distribution(candidate[a][:, hop_data.ANS_POS], native['full'][:, hop_data.ANS_POS], torch)} for a in R.ARMS}
            results[pop] = {'groups': groups, 'distribution_to_native': distributions, 'effects': effects,
                            'exact': exact, 'effect_vectors_close': effect_close, 'max_abs_replay_error': max_error,
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            print(json.dumps({'population': pop, 'x_only_query': distributions['X']['query'],
                              'high_repeated': groups['high_repeated'], 'hop1_novel': groups['hop1_novel']}), flush=True)
    pred_a = controls['passed'] and all(r['exact'] and r['groups']['high_repeated']['native_gold_probability'] >= .8 for r in results.values())
    pred_b = pred_a and all(m['passed'] for r in results.values() for m in r['distribution_to_native']['X'].values())
    pred_c = pred_a and all(r['groups'][g]['arms']['TT+CC']['gold_probability_loss'] >= .25
                            and abs(r['groups'][g]['arms']['X+CC']['gold_probability_loss']) <= .10
                            and abs(r['groups'][g]['arms']['TT+X']['gold_probability_loss']) <= .10
                            for r in results.values() for g in ('high_repeated', 'hop1_novel'))
    pred_d = pred_a and all(r['effect_vectors_close'] for r in results.values())
    receipt = {'experiment': 'hop_mlp_token_context_v1', 'runner_sha256': score.digest(SOURCE),
               'bound_sha256': EXPECTED, 'checkpoint_sha256': score.EXPECTED_CHECKPOINT, 'controls': controls,
               'populations': results, 'opaque_parameters_retained': sum(p.numel() for p in model.parameters()),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_mixed_sufficiency': bool(pred_b),
                               'pred_c_shared_mixed_dependency': bool(pred_c), 'pred_d_joint_interventions': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('shared_mixed_lookup_supported' if pred_b and pred_c else 'mixed_term_alone_not_supported'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import hop_mlp_terms_reference as R
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
