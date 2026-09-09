#!/usr/bin/env python3
"""Shared entity-equality routing, existing small contextual trained model.

pred_a_instrument: positive/perturbed/independent-router fixtures; native reference
atol=rtol=1e-9, unreduced lag-table logits atol=rtol=1e-4.
pred_b_distribution: each arm/population full and query KL mean<=1e-3, p99<=1e-2.
pred_c_interventions: centered-logit single/joint/interaction relL2<=.01;
if target RMS<1e-6, error RMS<=1e-8.
pred_d_live_reuse: both head removal effects RMS>1e-6; all prior gates;
physically removed Q/K constants exceed all replacement-table constants.
Null: token identity/asymmetric routing remains essential; do not retune this projection.
Price: 16 docs x3 populations x4 arms, batch4, T239, 29 full logits, 1800s watchdog;
each analysis allocation<256MiB, candidate bank37. No training. GPU via bqrunner.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_distribution pred_c_interventions pred_d_live_reuse
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language')
SOURCE = Path(__file__)
POLY = ROOT / 'basis_aligned/polynomial_causal'
OUT = POLY / 'EQUALITY_ROUTER_V1_RESULT.json'
PREREG = POLY / 'EQUALITY_ROUTER_V1_PREREGISTRATION.md'
REFERENCE = POLY / 'equality_router_reference.py'
CHECKPOINT = ROOT / 'runs_hop/attn-mlp-attn-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = '02a3f0e793849899629af95d88be4f60b5ff1b42ee39116cb2a7d8512cb212ff'
EXPECTED_REFERENCE = '0a235c2d529bba55c81d039e3e1706cdc57279c7a20a7e9a65fbb367052df19e'
EXPECTED_PREREG = 'd88a98435bc002c66e2034d8cad41a895e43de3617c14767e3fad5f4a7f00f1e'
ARMS = {'native': (), 'remove_h0': (0,), 'remove_h1': (1,), 'remove_joint': (0, 1)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(actual, expected, torch):
    lp, lq = expected.log_softmax(-1), actual.log_softmax(-1)
    kl = (lp.exp()*(lp-lq)).sum(-1)
    return dict(mean_kl=float(kl.mean()), p99_kl=float(torch.quantile(kl.flatten(), .99)),
                max_kl=float(kl.max()), max_abs_logit=float((actual-expected).abs().max()),
                top1_agreement=float((actual.argmax(-1) == expected.argmax(-1)).double().mean()),
                passed=bool(kl.mean() <= 1e-3 and torch.quantile(kl.flatten(), .99) <= 1e-2
                            and torch.isfinite(actual).all()))


def effect_error(actual, expected, torch):
    actual = actual-actual.mean(-1, keepdim=True)
    expected = expected-expected.mean(-1, keepdim=True)
    rms = float(expected.square().mean().sqrt())
    error = float((actual-expected).square().mean().sqrt())
    return dict(target_rms=rms, error_rms=error, relative_l2=error/max(rms, 1e-30),
                passed=bool((error <= 1e-8 if rms < 1e-6 else error/rms <= .01)
                            and torch.isfinite(actual).all()))


def datasets(torch, hop_data):
    iid, qa, qk = hop_data.sample_docs(16, torch.Generator().manual_seed(2909))
    g = torch.Generator().manual_seed(2910)
    names = torch.cat((torch.randperm(24, generator=g), torch.arange(24, 29)))
    renamed = names[iid]
    g = torch.Generator().manual_seed(2911)
    # Deliberately OOD function topology: all permutations, allowing short cycles.
    fmap = torch.rand(16, 24, generator=g).argsort(1)
    order = torch.rand(16, 24, generator=g).argsort(1)
    bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
    qe = torch.randint(24, (16, 48), generator=g)
    qko = torch.randint(4, (16, 48), generator=g)
    powers = hop_data._fpow(fmap, 3)
    qao = powers[torch.arange(16)[:, None], qko, qe]
    query = torch.stack((torch.full_like(qe, 24), qe, 25+qko, qao), -1).flatten(1)
    ood = torch.cat((bindings, query), 1)
    return {'iid': (iid, qa, qk), 'renaming_metamorphic': (renamed, names[qa], qk),
            'ood_short_cycles': (ood, qao, qko)}


def run(torch, R):
    import hop_data
    from hop_ablate import load
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runner selected but CUDA unavailable; no silent device change')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    model, cfg = load('attn-mlp-attn-rms-seed0')
    assert cfg['spec'] == ['attn', 'mlp', 'attn'] and cfg['norm'] == 'rms'
    model = model.to(device=device, dtype=torch.float64).eval()
    with torch.inference_mode():
        kernel = R.lag_kernel(model)
        coefficients = R.orbit_projection(kernel)
        ids = torch.arange(29, device=device)
        orbit = R.orbit_index(ids[:, None], ids[None, :])
        residual = kernel-coefficients[..., orbit]
        candidate = R.RouterProgram(model, coefficients, True).eval()
        table = R.RouterProgram(model, kernel, False).eval()
        candidate_constants = sum(p.numel() for p in candidate.parameters())+coefficients.numel()
        native_constants = sum(p.numel() for p in model.parameters())
        results = {}
        for population, (tokens, answers, hops) in datasets(torch, hop_data).items():
            all_native, all_candidate, all_table = {}, {}, {}
            exact_pass, table_pass = True, True
            exact_max, table_max = 0., 0.
            for name, remove in ARMS.items():
                lists = {k: [] for k in ('native', 'candidate', 'table')}
                for start in range(0, len(tokens), 4):
                    tok = tokens[start:start+4, :-1].to(device)
                    original = R.native_forward(model, tok, remove)
                    reference = R.reference_forward(model, tok, remove)
                    actual = candidate(tok, remove)
                    tabular = table(tok, remove)
                    exact_pass &= bool(torch.allclose(reference, original, atol=1e-9, rtol=1e-9))
                    table_pass &= bool(torch.allclose(tabular, original, atol=1e-4, rtol=1e-4))
                    exact_max = max(exact_max, float((reference-original).abs().max()))
                    table_max = max(table_max, float((tabular-original).abs().max()))
                    for key, value in (('native', original), ('candidate', actual), ('table', tabular)):
                        lists[key].append(value.cpu())
                all_native[name] = torch.cat(lists['native'])
                all_candidate[name] = torch.cat(lists['candidate'])
                all_table[name] = torch.cat(lists['table'])
            distribution_rows = {}
            for name in ARMS:
                distribution_rows[name] = {
                    'all_positions': distribution(all_candidate[name], all_native[name], torch),
                    'query_positions': distribution(all_candidate[name][:, hop_data.ANS_POS],
                                                    all_native[name][:, hop_data.ANS_POS], torch),
                    'lag_table_all_positions': distribution(all_table[name], all_native[name], torch),
                }
            effects = {}
            for name in ('remove_h0', 'remove_h1', 'remove_joint'):
                effects[name] = effect_error(all_candidate[name]-all_candidate['native'],
                                            all_native[name]-all_native['native'], torch)
            def interaction(arms):
                return arms['remove_joint']-arms['remove_h0']-arms['remove_h1']+arms['native']
            effects['interaction'] = effect_error(interaction(all_candidate), interaction(all_native), torch)
            results[population] = dict(
                token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                rows=16, input_positions=239, exact_reference_passed=exact_pass,
                exact_reference_max_abs=exact_max, lag_table_passed=table_pass,
                lag_table_max_abs=table_max, distributions=distribution_rows, effects=effects,
                native_accuracy_by_hop=hop_data.score_by_hop(all_native['native'], answers, hops),
                candidate_accuracy_by_hop=hop_data.score_by_hop(all_candidate['native'], answers, hops))
            print(json.dumps({'population': population, 'reference_pass': exact_pass,
                              'lag_table_pass': table_pass,
                              'native_arm': distribution_rows['native'], 'effects': effects}), flush=True)
        fixtures = R.control_fixtures()
        pred_a = fixtures['passed'] and all(r['exact_reference_passed'] and r['lag_table_passed']
                                          for r in results.values())
        pred_b = all(d[scope]['passed'] for r in results.values() for d in r['distributions'].values()
                     for scope in ('all_positions', 'query_positions'))
        pred_c = all(e['passed'] for r in results.values() for e in r['effects'].values())
        live = all(r['effects'][a]['target_rms'] > 1e-6 for r in results.values()
                   for a in ('remove_h0', 'remove_h1'))
        pred_d = pred_a and pred_b and pred_c and live and candidate_constants < native_constants
        torch.cuda.synchronize()
        # Store the literal discovered coefficients; all native background is checkpoint-bound.
        artifact = POLY / 'EQUALITY_ROUTER_V1_COEFFICIENTS.pt'
        torch.save({'coefficients': coefficients.cpu(), 'checkpoint_sha256': EXPECTED_CHECKPOINT}, artifact)
        result = dict(
            predictions={'pred_a_instrument': pred_a, 'pred_b_distribution': pred_b,
                         'pred_c_interventions': pred_c, 'pred_d_live_reuse': pred_d},
            terminal=('candidate_passes_small_model_screen' if pred_d else
                      'invariant_router_rejected' if pred_a else 'instrument_invalid'),
            fixtures=fixtures, populations=results, live_removals=live,
            kernel_residual_relative_l2=float(residual.norm()/kernel.norm()),
            kernel_residual_per_head=[float(residual[h].norm()/kernel[h].norm()) for h in range(4)],
            constants=dict(native_parameters=native_constants, candidate_parameters_and_table=candidate_constants,
                           retained_native_parameters=sum(p.numel() for p in candidate.parameters()),
                           learned_table=coefficients.numel(), removed_qk=4*128*128,
                           native_fixed_buffers=sum(b.numel() for b in model.buffers()),
                           candidate_fixed_buffers_excluding_table=sum(b.numel() for b in candidate.buffers())-coefficients.numel(),
                           full_table_baseline=kernel.numel()),
            scope='First-layer token-input router only; embeddings, values, output map and full contextual suffix retained and charged. No bilin18 circuit claim.',
            execution=dict(device=str(torch.cuda.get_device_name()), torch=torch.__version__, dtype='float64',
                           seconds=time.perf_counter()-started, max_kernel_bytes=kernel.numel()*kernel.element_size(),
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),
            hashes={str(p.relative_to(ROOT)): digest(p) for p in
                    (CHECKPOINT, PREREG, REFERENCE, SOURCE, artifact)})
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'terminal': result['terminal'], 'predictions': result['predictions'],
                          'seconds': result['execution']['seconds']}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert digest(REFERENCE) == EXPECTED_REFERENCE, 'reference hash mismatch'
    assert digest(PREREG) == EXPECTED_PREREG, 'preregistration hash mismatch'
    import torch
    import equality_router_reference as R
    checks = R.control_fixtures()
    assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'fixtures': checks, 'bank_columns': 37,
                          'kernel_bytes': 4*240*29*29*8, 'max_seconds': 1800}))
        return
    run(torch, R)


if __name__ == '__main__':
    main()
