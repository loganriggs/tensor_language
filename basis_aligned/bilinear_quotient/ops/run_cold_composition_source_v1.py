#!/usr/bin/env python3
"""Cold multi-hop final-source test on an existing composition-capable checkpoint.

pred_a_instrument: parser/native/fold all5 arms1e-9; each native hop1..3 acc>=.8.
pred_b_answer_binding_sufficient: keepTQ query29-way KL mean<=1e-3,p99<=1e-2.
pred_c_selective_target: removeT gold loss>=.25, abs removeP<=.10, each hop2/3.
pred_d_joint_extraction: centered removal vectors and disjoint joint identity1e-9.
Null: inspect composed information in earlier binding states, no head/rank sweep.
128 independent worlds x4 query forks x5arms,B4 FP64,1800s,<256MiB/tensor; managed GPU.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_answer_binding_sufficient pred_c_selective_target pred_d_joint_extraction
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
OUT = POLY/'COLD_COMPOSITION_SOURCE_V1_RESULT.json'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'cold_composition_source_reference.py', POLY/'contextual_history_reference.py',
         POLY/'COLD_COMPOSITION_SOURCE_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '6b292fa34d35a92a689c8137705ec5a77a2d1fd5f4d4151a6fecc19538d933c1'
ARMS = ('full', 'removeT', 'removeP', 'removeTP', 'keepTQ')


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch, data):
    out = {}
    for kind, seed in (('iid', 10909), ('ood_short_cycles', 10910)):
        g = torch.Generator().manual_seed(seed)
        perm = torch.rand(64, 24, generator=g).argsort(1)
        fmap = torch.empty_like(perm).scatter_(1, perm, perm.roll(-1, 1)) if kind == 'iid' else perm
        order = torch.rand(64, 24, generator=g).argsort(1)
        bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
        entity = torch.randint(24, (64,), generator=g)
        hops = torch.arange(4).repeat(64)
        qe = entity.repeat_interleave(4)
        query = torch.stack((torch.full_like(qe, 24), qe, 25+hops), -1)
        tok = torch.cat((bindings.repeat_interleave(4, 0), query), 1)
        answers = data._fpow(fmap, 3)[torch.arange(64).repeat_interleave(4), hops, qe]
        out[kind] = tok, answers, hops
    return out


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
        for pop, (tokens, answers, hops) in populations(torch, hop_data).items():
            native = {a: [] for a in ARMS}; candidate = {a: [] for a in ARMS}
            exact = True; max_error = 0.; source_counts = []
            for i in range(0, len(tokens), 4):
                masks = {k: v.cuda() for k, v in R.source_masks(tokens[i:i+4]).items()}
                source_counts.extend(torch.stack([v.sum((1, 2)) for v in masks.values()], -1).cpu().tolist())
                tok = tokens[i:i+4].cuda()
                parts = program.parts(tok, masks)
                full = parts['R']+parts['H']+parts['B']+parts['O']
                outputs = {'full': full, 'removeT': full-parts['H'], 'removeP': full-parts['B'],
                           'removeTP': full-parts['H']-parts['B'], 'keepTQ': full.clone()}
                outputs['keepTQ'][:, 50] = (parts['R']+parts['H']+parts['C'])[:, 50]
                other = torch.zeros_like(masks['H']); other[:, 50] = ~(masks['H'] | masks['C'])[:, 50]
                cuts = {'full': None, 'removeT': masks['H'], 'removeP': masks['B'],
                        'removeTP': masks['H'] | masks['B'], 'keepTQ': other}
                for a in ARMS:
                    n = R.native_forward(model, tok, cuts[a]); c = outputs[a]
                    exact &= bool(torch.allclose(n, c, atol=1e-9, rtol=1e-9))
                    max_error = max(max_error, float((n-c).abs().max()))
                    native[a].append(n.cpu()); candidate[a].append(c.cpu())
            native = {a: torch.cat(v) for a, v in native.items()}; candidate = {a: torch.cat(v) for a, v in candidate.items()}
            probs = {a: v[:, 50].softmax(-1).gather(-1, answers[:, None]).squeeze(-1) for a, v in native.items()}
            groups = {}
            for k in range(4):
                select = hops == k
                groups[str(k)] = {'n_worlds': int(select.sum()),
                    'arms': {a: {'accuracy': float((v[:, 50].argmax(-1) == answers)[select].double().mean()),
                                 'gold_probability': float(probs[a][select].mean()),
                                 'gold_probability_loss': float((probs['full']-probs[a])[select].mean())} for a, v in native.items()},
                    'keepTQ_distribution': score.distribution(candidate['keepTQ'][select, 50], native['full'][select, 50], torch)}
            def center(x):
                return x-x.mean(-1, keepdim=True)
            effect_close = all(torch.allclose(center(candidate[a]-candidate['full']), center(native[a]-native['full']), atol=1e-9, rtol=1e-9) for a in ARMS)
            joint = native['removeT']+native['removeP']-native['full']
            joint_close = bool(torch.allclose(joint, native['removeTP'], atol=1e-9, rtol=1e-9))
            results[pop] = {'groups': groups, 'exact': exact, 'max_abs_replay_error': max_error,
                            'effect_vectors_close': effect_close, 'joint_close': joint_close,
                            'joint_max_abs': float((joint-native['removeTP']).abs().max()), 'source_counts_T_P_Q': source_counts,
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            print(json.dumps({'population': pop, 'groups': groups, 'replay_max_abs': max_error}), flush=True)
    pred_a = controls['passed'] and all(r['exact'] and all(r['groups'][str(k)]['arms']['full']['accuracy'] >= .8 for k in (1, 2, 3)) for r in results.values())
    pred_b = pred_a and all(r['groups'][str(k)]['keepTQ_distribution']['passed'] for r in results.values() for k in (1, 2, 3))
    pred_c = pred_a and all(r['groups'][str(k)]['arms']['removeT']['gold_probability_loss'] >= .25
                            and abs(r['groups'][str(k)]['arms']['removeP']['gold_probability_loss']) <= .10 for r in results.values() for k in (2, 3))
    pred_d = pred_a and all(r['effect_vectors_close'] and r['joint_close'] for r in results.values())
    receipt = {'experiment': 'cold_composition_source_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': controls, 'populations': results,
               'independent_worlds': 128, 'correlated_query_forks': 512,
               'compiled_opaque_constants': sum(p.numel() for p in program.parameters())+program.folded.numel(),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_answer_binding_sufficient': bool(pred_b),
                               'pred_c_selective_target': bool(pred_c), 'pred_d_joint_extraction': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('direct_answer_binding_supported' if pred_b and pred_c else 'direct_answer_binding_not_supported'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import cold_composition_source_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls(); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
