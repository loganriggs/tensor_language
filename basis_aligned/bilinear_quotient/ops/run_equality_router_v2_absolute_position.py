#!/usr/bin/env python3
"""Absolute-position repair for v1; same fixed symmetry hypothesis and populations.

pred_a_instrument: native absolute-position kernel and full forward close at1e-9.
pred_b_distribution: all arms/populations KL mean<=1e-3, p99<=1e-2.
pred_c_interventions: full centered single/joint/interaction errors <=.01 relative,
or <=1e-8 absolute RMS for target RMS<1e-6.
pred_d_same_fixed_hypothesis: absolute and lag-projected programs close at1e-4.
Null: first-layer invariant rule fails, now without lag-reassociation contamination.
Price: 239 slabs<7MiB, diagnostic table<68MiB, no allocation>256MiB, 1800 seconds.
GPU via bqrunner. Not a new held-out dataset or a compact compiled artifact.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_distribution pred_c_interventions pred_d_same_fixed_hypothesis
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
REF = POLY/'equality_router_reference.py'
PARENT = ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py'
PREREG = POLY/'EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_PREREGISTRATION.md'
OUT = POLY/'EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_RESULT.json'
EXPECTED = 'a4afc394dc31e0aa09ebd2e3a0a2a579d56bd5800cd666de5a1d975b7d56c554'


def bound_hash():
    return hashlib.sha256(REF.read_bytes()+PARENT.read_bytes()+PREREG.read_bytes()).hexdigest()


def run(torch, R, prior):
    from hop_ablate import load
    import hop_data
    import types
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert prior.digest(prior.CHECKPOINT) == prior.EXPECTED_CHECKPOINT
    model, _cfg = load('attn-mlp-attn-rms-seed0')
    model = model.to(device='cuda', dtype=torch.float64).eval()
    populations = prior.datasets(torch, hop_data)
    with torch.inference_mode():
        tokens = torch.cat([p[0][:, :-1] for p in populations.values()]).cuda()
        length = tokens.shape[1]
        layer = model.layers[0]
        n = layer.norm(model.embed.weight)
        def factors(name):
            z = getattr(layer, name)(n).reshape(29, 4, 32)
            left, right = z.chunk(2, -1)
            c = layer.rotary.cos_cached[0, :length, 0][:, None, None]
            s = layer.rotary.sin_cached[0, :length, 0][:, None, None]
            return z[None]*c + torch.cat((-right, left), -1)[None]*s
        q1, k1, q2, k2 = [factors(name) for name in ('q1', 'k1', 'q2', 'k2')]
        coeff = torch.empty(4, length, length, 37, device='cuda', dtype=torch.float64)
        observed = torch.empty(len(tokens), 4, length, length, device='cuda', dtype=torch.float64)
        head_ids = torch.arange(4, device='cuda')[None, :, None]
        source_pos = torch.arange(length, device='cuda')[None, None]
        for t in range(length):
            slab = (torch.einsum('ahi,sbhi->hsab', q1[t], k1)
                    *torch.einsum('ahi,sbhi->hsab', q2[t], k2))/(32*32)
            coeff[:, t] = R.orbit_projection(slab)
            observed[:, :, t] = slab[head_ids, source_pos,
                                     tokens[:, t, None, None], tokens[:, None, :]]
        causal = torch.ones(length, length, device='cuda', dtype=torch.bool).tril()
        observed.masked_fill_(~causal[None, None], 0)
        # Native pattern is gathered in batches to keep every tensor <256MiB.
        native = torch.cat([layer.pattern(model.embed(tokens[i:i+4])) for i in range(0, len(tokens), 4)])
        pattern_pass = bool(torch.allclose(observed, native, atol=1e-9, rtol=1e-9))
        pattern_error = float((observed-native).abs().max())
        del observed, native
        absolute = R.RouterProgram(model, coeff, True)
        def absolute_pattern(self, tok):
            pos = torch.arange(tok.shape[1], device=tok.device)
            h = torch.arange(4, device=tok.device)[None, :, None, None]
            oi = R.orbit_index(tok[:, None, :, None], tok[:, None, None, :])
            scores = self.table[h, pos[None, None, :, None], pos[None, None, None, :], oi]
            return scores.masked_fill(pos[None, None, :, None] < pos[None, None, None, :], 0)
        absolute.pattern = types.MethodType(absolute_pattern, absolute)
        lag_coeff = R.orbit_projection(R.lag_kernel(model))
        lag = R.RouterProgram(model, lag_coeff, True)
        results = {}
        for population, (tok, answers, hops) in populations.items():
            outputs = {method: {} for method in ('native', 'absolute', 'lag')}
            exact_pass, same_pass = True, True
            exact_max, same_max = 0., 0.
            for arm, remove in prior.ARMS.items():
                pieces = {method: [] for method in outputs}
                for i in range(0, len(tok), 4):
                    batch = tok[i:i+4, :-1].cuda()
                    expected = R.native_forward(model, batch, remove)
                    ref = R.reference_forward(model, batch, remove)
                    a, l = absolute(batch, remove), lag(batch, remove)
                    exact_pass &= bool(torch.allclose(ref, expected, atol=1e-9, rtol=1e-9))
                    same_pass &= bool(torch.allclose(a, l, atol=1e-4, rtol=1e-4))
                    exact_max = max(exact_max, float((ref-expected).abs().max()))
                    same_max = max(same_max, float((a-l).abs().max()))
                    for name, data in (('native', expected), ('absolute', a), ('lag', l)):
                        pieces[name].append(data.cpu())
                for method in outputs:
                    outputs[method][arm] = torch.cat(pieces[method])
            nout, aout = outputs['native'], outputs['absolute']
            distributions = {a: {s: prior.distribution(aout[a][:, idx], nout[a][:, idx], torch)
                                  for s, idx in (('all_positions', slice(None)), ('query_positions', hop_data.ANS_POS))}
                             for a in prior.ARMS}
            effects = {a: prior.effect_error(aout[a]-aout['native'], nout[a]-nout['native'], torch)
                       for a in ('remove_h0', 'remove_h1', 'remove_joint')}
            def interaction(out):
                return out['remove_joint']-out['remove_h0']-out['remove_h1']+out['native']
            effects['interaction'] = prior.effect_error(interaction(aout), interaction(nout), torch)
            results[population] = dict(exact_reference_passed=exact_pass, exact_reference_max_abs=exact_max,
                                       absolute_vs_lag_passed=same_pass, absolute_vs_lag_max_abs=same_max,
                                       distributions=distributions, effects=effects)
            print(json.dumps({'population': population, 'query_native_arm': distributions['native']['query_positions'],
                              'exact_reference': exact_pass, 'same_hypothesis': same_pass}), flush=True)
        pred_a = pattern_pass and all(r['exact_reference_passed'] for r in results.values())
        pred_b = all(s['passed'] for r in results.values() for a in r['distributions'].values() for s in a.values())
        pred_c = all(s['passed'] for r in results.values() for s in r['effects'].values())
        pred_d = all(r['absolute_vs_lag_passed'] for r in results.values())
        torch.cuda.synchronize()
        result = dict(predictions={'pred_a_instrument': pred_a, 'pred_b_distribution': pred_b,
                                  'pred_c_interventions': pred_c, 'pred_d_same_fixed_hypothesis': pred_d},
                      terminal=('instrument_invalid' if not pred_a else
                                'invariant_router_rejected_with_absolute_positions' if not(pred_b and pred_c) else
                                'absolute_position_router_passes_fidelity_only'),
                      populations=results, native_pattern_max_abs=pattern_error,
                      diagnostic_table_constants=coeff.numel(), diagnostic_table_bytes=coeff.numel()*8,
                      compact_candidate=False, seconds=time.perf_counter()-started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      evidence_scope='Reused v1 diagnostic populations; exact absolute-position repair, not held-out confirmation.',
                      hashes={str(p.relative_to(ROOT)): prior.digest(p) for p in
                              (SOURCE, REF, PARENT, PREREG, prior.CHECKPOINT, prior.OUT)})
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'terminal': result['terminal'], 'predictions': result['predictions'],
                          'seconds': result['seconds']}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import equality_router_reference as R
    import run_equality_router_v1 as prior
    assert R.control_fixtures()['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'table_bytes': 4*239*239*37*8,
                          'slab_bytes': 4*239*29*29*8, 'checkpoint_opened': False}))
        return
    run(torch, R, prior)


if __name__ == '__main__':
    main()
