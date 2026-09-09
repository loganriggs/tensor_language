#!/usr/bin/env python3
"""First-layer joint read-route-write with self and previous-token sources.

pred_a_instrument: dense-restricted versus two-shift full logits atol=rtol=1e-9.
pred_b_distribution: all populations/arms, full/query KL mean<=1e-3 and p99<=1e-2.
pred_c_interventions: single/joint/interaction centered errors<=.01 relative;
error RMS<=1e-8 when target RMS<1e-6.
pred_d_live_branches: both native branch-removal effects RMS>1e-6.
Null: radius-one locality fails; no radius or threshold sweep.
Price: unchanged400640 parameters; first-layer edges T(T+1)/2 ->2T-1;
batch4,16docs x3populations x4arms, T239, 1800s. GPU only via managed runner.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_distribution pred_c_interventions pred_d_live_branches
from __future__ import annotations
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
REF = POLY/'local_transport_reference.py'
PREREG = POLY/'LOCAL_TRANSPORT_V1_PREREGISTRATION.md'
SCORER = ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py'
OUT = POLY/'LOCAL_TRANSPORT_V1_RESULT.json'
EXPECTED = 'bc23ec261a44df8464e42ee43b2bf80ddc7b5e3035f44c50bb805fe796b4a88f'
ARMS = {'native': (), 'remove_self': (0,), 'remove_previous': (1,), 'remove_joint': (0, 1)}


def bound_hash():
    return hashlib.sha256(REF.read_bytes()+PREREG.read_bytes()+SCORER.read_bytes()).hexdigest()


def populations(torch, data):
    iid, qa, qk = data.sample_docs(16, torch.Generator().manual_seed(3909))
    g = torch.Generator().manual_seed(3910)
    names = torch.cat((torch.randperm(24, generator=g), torch.arange(24, 29)))
    g = torch.Generator().manual_seed(3911)
    fmap = torch.rand(16, 24, generator=g).argsort(1)
    order = torch.rand(16, 24, generator=g).argsort(1)
    bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
    qe = torch.randint(24, (16, 48), generator=g)
    qko = torch.randint(4, (16, 48), generator=g)
    powers = data._fpow(fmap, 3)
    qao = powers[torch.arange(16)[:, None], qko, qe]
    blocks = torch.stack((torch.full_like(qe, 24), qe, 25+qko, qao), -1).flatten(1)
    return {'iid': (iid, qa, qk), 'renaming_metamorphic': (names[iid], names[qa], qk),
            'ood_short_cycles': (torch.cat((bindings, blocks), 1), qao, qko)}


def run(torch, R, score):
    from hop_ablate import load
    import hop_data
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert score.digest(score.CHECKPOINT) == score.EXPECTED_CHECKPOINT
    model, cfg = load('attn-mlp-attn-rms-seed0')
    assert cfg['spec'] == ['attn', 'mlp', 'attn']
    model = model.to(device='cuda', dtype=torch.float64).eval()
    program = R.LocalTransport(model).eval()
    results = {}
    with torch.inference_mode():
        for population, (tokens, answers, hops) in populations(torch, hop_data).items():
            native, actual = {}, {}
            replay_pass, replay_max = True, 0.
            for arm, remove in ARMS.items():
                ns, cs = [], []
                for i in range(0, len(tokens), 4):
                    tok = tokens[i:i+4, :-1].cuda()
                    n = R.dense_forward(model, tok, remove)
                    a = program(tok, remove)
                    r = R.dense_forward(model, tok, remove, restrict=True)
                    replay_pass &= bool(torch.allclose(a, r, atol=1e-9, rtol=1e-9))
                    replay_max = max(replay_max, float((a-r).abs().max()))
                    ns.append(n.cpu()); cs.append(a.cpu())
                native[arm], actual[arm] = torch.cat(ns), torch.cat(cs)
            ds = {a: {s: score.distribution(actual[a][:, idx], native[a][:, idx], torch)
                      for s, idx in (('all_positions', slice(None)), ('query_positions', hop_data.ANS_POS))}
                  for a in ARMS}
            es = {a: score.effect_error(actual[a]-actual['native'], native[a]-native['native'], torch)
                  for a in ('remove_self', 'remove_previous', 'remove_joint')}
            def interaction(out):
                return out['remove_joint']-out['remove_self']-out['remove_previous']+out['native']
            es['interaction'] = score.effect_error(interaction(actual), interaction(native), torch)
            results[population] = dict(distributions=ds, effects=es,
                                       restricted_replay_passed=replay_pass, restricted_replay_max_abs=replay_max,
                                       token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                                       native_accuracy_by_hop=hop_data.score_by_hop(native['native'], answers, hops),
                                       compiled_accuracy_by_hop=hop_data.score_by_hop(actual['native'], answers, hops))
            print(json.dumps({'population': population, 'reference_pass': replay_pass,
                              'query_distribution': ds['native']['query_positions'], 'effects': es}), flush=True)
        controls = R.controls()
        pred_a = controls['passed'] and all(r['restricted_replay_passed'] for r in results.values())
        pred_b = all(d['passed'] for r in results.values() for a in r['distributions'].values() for d in a.values())
        pred_c = all(e['passed'] for r in results.values() for e in r['effects'].values())
        pred_d = all(r['effects'][a]['target_rms'] > 1e-6 for r in results.values()
                     for a in ('remove_self', 'remove_previous'))
        torch.cuda.synchronize()
        result = dict(predictions={'pred_a_instrument': pred_a, 'pred_b_distribution': pred_b,
                                  'pred_c_interventions': pred_c, 'pred_d_live_branches': pred_d},
                      terminal=('instrument_invalid' if not pred_a else 'local_transport_screen_passes' if
                                pred_b and pred_c and pred_d else 'radius_one_transport_rejected'),
                      populations=results, controls=controls,
                      parameters=sum(p.numel() for p in program.parameters()),
                      first_layer_edges_per_head={'native': 239*240//2, 'compiled': 2*239-1},
                      fixed_buffer_scalars=sum(b.numel() for b in program.buffers()),
                      scope='First-layer dependency simplification only; full native opaque weights and contextual background retained. No bilin18 or complete circuit-discovery claim.',
                      seconds=time.perf_counter()-started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      hashes={str(p.relative_to(ROOT)): score.digest(p) for p in
                              (REF, PREREG, SCORER, SOURCE, score.CHECKPOINT)})
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'terminal': result['terminal'], 'predictions': result['predictions']}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import local_transport_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2)
    checks = R.controls()
    assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False}))
        return
    run(torch, R, score)


if __name__ == '__main__':
    main()
